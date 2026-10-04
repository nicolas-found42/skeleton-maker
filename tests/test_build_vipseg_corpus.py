# SPDX-License-Identifier: MIT
"""The VIPSeg converter, checked on a 20x16 synthetic dataset whose labels are known exactly.

Every expectation is worked out from the mask encoding (``0`` unlabelled, below 125 a stuff class
with ``id = value - 1``, otherwise a thing with ``id = value // 100 - 1``), never from the
converter's own mapping.
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from skeleton_maker.envannotations import load_annotation, load_mask

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_vipseg_corpus.py"
_spec = importlib.util.spec_from_file_location("build_vipseg_corpus", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
vip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vip)

W, H = 20, 16
FRAMES = 14
STRIDE = 3
FIRST = 100
CATEGORIES = [
    {"id": 0, "name": "wall", "isthing": 0},
    {"id": 1, "name": "ceiling", "isthing": 0},
    {"id": 2, "name": "door", "isthing": 1},
    {"id": 13, "name": "floor", "isthing": 0},
    {"id": 18, "name": "road", "isthing": 0},
    {"id": 28, "name": "sky", "isthing": 0},
    {"id": 75, "name": "cupboard_or_showcase_or_storage_rack", "isthing": 0},
    {"id": 48, "name": "car", "isthing": 1},
    {"id": 60, "name": "person", "isthing": 1},
    {"id": 89, "name": "table_or_desk", "isthing": 1},
]
WALL, CEILING, FLOOR, CUPBOARD = 1, 2, 14, 76
CAR = 49 * 100
PERSON = 61 * 100
TABLE = 90 * 100


def _mask(i: int) -> np.ndarray:
    """Frame ``i``: a ceiling for the first seven frames, a car sliding right one pixel a frame."""
    m = np.full((H, W), WALL, np.uint16)
    m[12:, :] = FLOOR
    if i < 7:
        m[:3, :] = CEILING
    m[5:8, 12:15] = TABLE
    m[8:11, i : i + 3] = CAR
    if i >= 5:
        m[8:11, i + 1] = CUPBOARD  # a post in front of the car's middle splits it in two
    m[3:6, 0:2] = PERSON
    m[6:8, 16:18] = CUPBOARD
    m[0:1, 18:20] = 0  # unlabelled
    return m


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "VIPSeg"
    (root / "VIPSeg_720P").mkdir(parents=True)
    (root / "VIPSeg_720P" / "panoVIPSeg_categories.json").write_text(json.dumps(CATEGORIES))
    rng = np.random.default_rng(3)
    for sub in ("imgs", "panomasks"):
        (root / sub / "v1").mkdir(parents=True)
    for i in range(FRAMES):
        number = f"{FIRST + STRIDE * i:08d}"
        image = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(root / "imgs" / "v1" / f"{number}.jpg"), image)
        cv2.imwrite(str(root / "panomasks" / "v1" / f"{number}.png"), _mask(i))
    return root


def _build(dataset, tmp_path, **kwargs):
    kwargs.setdefault("shots_override", [(0, FRAMES - 1)])
    kwargs.setdefault("shift_override", 5.0)
    return vip.build(
        dataset,
        "v1",
        tmp_path / "out",
        name="clip",
        split="heldout",
        encode=False,
        clip_sha256="b" * 64,
        **kwargs,
    )


def test_decode_reads_stuff_and_thing_values():
    category, instance = vip.decode(np.array([[0, 1, 49 * 100 + 3]], np.uint16))

    assert category.tolist() == [[-1, 0, 48]]
    assert instance.tolist() == [[-1, -1, 3]]


def test_things_become_vehicles_or_objects_and_people_are_never_instances():
    cats = {c["id"]: c for c in CATEGORIES}

    assert vip.family_of(48, cats) == "vehicle"
    assert vip.family_of(89, cats) == "object"
    assert vip.family_of(60, cats) is None
    assert vip.family_of(0, cats) is None
    assert vip.is_ignored(60, cats)
    assert vip.family_of(2, cats) is None  # a door is a surface to the scan
    assert vip.is_ignored(2, cats)
    assert not vip.is_ignored(48, cats)


def test_the_annotation_loads_and_names_its_provenance(dataset, tmp_path):
    result = _build(dataset, tmp_path)

    loaded = load_annotation(tmp_path / "out" / "clip.json")
    assert loaded["scope"] == "semantic"
    assert loaded["provenance"]["kind"] == "published_dataset"
    assert len(loaded["frames"]) == FRAMES
    assert result["report"]["source_frames"][:2] == [FIRST, FIRST + STRIDE]


def test_surfaces_instances_and_ignore_regions_follow_the_masks(dataset, tmp_path):
    _build(dataset, tmp_path)
    annotation = load_annotation(tmp_path / "out" / "clip.json")
    frame = annotation["frames"][0]

    surfaces = {s["class"]: load_mask(annotation, s["mask"], (H, W)) for s in frame["surfaces"]}
    assert set(surfaces) == {"wall", "ceiling", "floor"}
    assert surfaces["floor"][12:, :].all()
    assert surfaces["floor"].sum() == 4 * W
    assert surfaces["ceiling"].sum() == 3 * W - 2  # two pixels are unlabelled
    instances = {i["id"]: i for i in frame["instances"]}
    assert set(instances) == {"car-0", "table_or_desk-0"}
    assert instances["car-0"]["family"] == "vehicle"
    car = load_mask(annotation, instances["car-0"]["mask"], (H, W))
    assert car.sum() == 9
    assert car[8:11, 0:3].all()
    ignore = load_mask(annotation, frame["ignore"][0]["mask"], (H, W))
    expected = np.zeros((H, W), bool)
    expected[3:6, 0:2] = True  # person
    expected[6:8, 16:18] = True  # cupboard: a stuff class a detector may report as an object
    expected[0:1, 18:20] = True  # unlabelled
    assert (ignore == expected).all()


def test_a_class_absent_from_a_well_labelled_frame_is_class_negative(dataset, tmp_path):
    _build(dataset, tmp_path)
    annotation = load_annotation(tmp_path / "out" / "clip.json")

    assert annotation["frames"][0]["negative_classes"] == []
    assert annotation["frames"][10]["negative_classes"] == ["ceiling"]


def test_a_badly_labelled_frame_makes_no_negative_claim(dataset, tmp_path):
    for path in sorted((dataset / "panomasks" / "v1").glob("*.png"))[10:]:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        mask[:, 4:] = 0  # most of the frame unlabelled
        cv2.imwrite(str(path), mask)
    _build(dataset, tmp_path)
    annotation = load_annotation(tmp_path / "out" / "clip.json")

    assert annotation["frames"][10]["negative_classes"] == []


def test_the_tracking_interval_stays_inside_one_shot_and_spans_two_seconds(dataset, tmp_path):
    result = _build(dataset, tmp_path)
    assert result["annotation"]["tracking_intervals"] == [{"first_frame": 0, "last_frame": 13}]

    with pytest.raises(vip.ConversionError, match="two-second"):
        _build(dataset, tmp_path, shots_override=[(0, 6), (7, 13)])


def test_unevenly_spaced_frames_are_refused(dataset, tmp_path):
    victim = sorted((dataset / "imgs" / "v1").glob("*.jpg"))[5]
    victim.rename(victim.with_name("00000999.jpg"))
    mask = sorted((dataset / "panomasks" / "v1").glob("*.png"))[5]
    mask.rename(mask.with_name("00000999.png"))

    with pytest.raises(vip.ConversionError, match="evenly spaced"):
        _build(dataset, tmp_path)


def test_a_clip_with_a_fixed_camera_is_tagged_from_what_was_measured(dataset, tmp_path):
    result = _build(dataset, tmp_path, shift_override=0.0)

    tags = set(result["annotation"]["tags"])
    assert {"tripod", "underconstrained", "vehicle_moving", "occlusion"} <= tags
    assert "person_free" not in tags
    assert "translating" not in tags
    assert "motion_blur" in result["report"]["measured"]["not_measured"]


def test_a_moving_camera_gets_no_tripod_or_vehicle_motion_tags(dataset, tmp_path):
    result = _build(dataset, tmp_path, shift_override=5.0)

    tags = set(result["annotation"]["tags"])
    assert tags.isdisjoint({"tripod", "underconstrained", "vehicle_moving", "vehicle_stationary"})


def test_a_cut_is_an_edited_cut_and_a_clip_with_a_ceiling_is_indoor(dataset, tmp_path):
    result = _build(dataset, tmp_path, shots_override=[(0, 2), (3, 13)])

    tags = set(result["annotation"]["tags"])
    assert {"edited_cut", "indoor"} <= tags
    assert "outdoor" not in tags
    assert result["annotation"]["tracking_intervals"] == [{"first_frame": 3, "last_frame": 13}]


def test_mask_only_statistics_summarise_a_video(dataset):
    cats = vip.load_categories(dataset)

    stats = vip.class_stats(dataset, "v1", cats)

    assert stats["frames"] == FRAMES
    assert stats["strides"] == [STRIDE]
    assert stats["surface_frames"] == {"wall": FRAMES, "ceiling": 7, "floor": FRAMES}
    assert stats["vehicle_instances"] == {"car": 1}
    assert stats["object_classes"] == ["table_or_desk"]
    assert stats["person_frames"] == FRAMES
    assert stats["even_size"] is True
