# SPDX-License-Identifier: MIT
"""The accuracy scorer, checked against corpora whose scores were worked out by hand.

Every expected number below comes from counting pixels or instances on a 10x10 frame, never
from running the scorer's own formulas.
"""

import json
import math

import pytest

from skeleton_maker import cli, envscore

from .env_corpus import Corpus

FULL = (0, 10, 0, 10)


def _score(corpus, tmp_path, *extra):
    out = tmp_path / "score.json"
    rc = cli.main(
        [
            "environment-score",
            "--predictions",
            str(corpus.predictions),
            "--annotations",
            str(corpus.annotations),
            "--out",
            str(out),
            *extra,
        ]
    )
    return rc, json.loads(out.read_text())


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path)


def _px(i, row=0):
    """A one-pixel rectangle in column i."""
    return (row, row + 1, i, i + 1)


# --- objects and vehicles ------------------------------------------------------------


def test_instance_matching_counts_true_false_positives_and_misses(corpus, tmp_path):
    a = corpus.clip()
    a.gt_instance(0, "chair-1", "object", "chair", (0, 4, 0, 4))  # 16 px
    a.gt_instance(0, "chair-2", "object", "chair", (5, 9, 5, 9))  # 16 px
    a.gt_instance(0, "table-1", "object", "table", (8, 10, 0, 3))  # 6 px, never predicted
    a.pred(0, "e1", "object", "chair", (0, 4, 0, 4))  # IoU 1.0
    a.pred(0, "e2", "object", "chair", (5, 8, 5, 9))  # 12 px inside 16: IoU 0.75
    a.pred(0, "e3", "object", "lamp", (0, 2, 8, 10))  # nothing there
    corpus.write()

    rc, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["objects"]
    assert gate["counts"] == {"tp": 2, "fp": 1, "fn": 1}
    assert gate["precision"] == pytest.approx(2 / 3)
    assert gate["recall"] == pytest.approx(2 / 3)
    assert gate["per_class"]["chair"]["tp"] == 2
    assert gate["per_class"]["lamp"]["fp"] == 1
    assert gate["per_class"]["table"]["fn"] == 1
    assert gate["pass"] is False
    assert doc["gates"]["vehicles"]["pass"] is False
    assert rc == 1


def test_a_gate_can_pass_while_others_fail(corpus, tmp_path):
    a = corpus.clip()
    for i in range(10):
        a.gt_instance(0, f"o{i}", "object", "chair", _px(i))
    for i in range(8):  # 8 matches
        a.pred(0, f"e{i}", "object", "chair", _px(i))
    a.pred(0, "stray", "object", "chair", _px(0, row=5))  # 1 false positive; o8, o9 missed
    corpus.write()

    rc, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["objects"]
    assert gate["counts"] == {"tp": 8, "fp": 1, "fn": 2}
    assert gate["precision"] == pytest.approx(8 / 9)  # >= 0.80
    assert gate["recall"] == pytest.approx(0.8)  # >= 0.70
    assert gate["pass"] is True
    assert doc["gates"]["ceilings"]["pass"] is False
    assert doc["all_gates_passed"] is False
    assert rc == 1


def test_wrong_class_is_both_a_false_positive_and_a_miss(corpus, tmp_path):
    a = corpus.clip()
    a.gt_instance(0, "t", "object", "table", (0, 4, 0, 4))
    a.pred(0, "e", "object", "chair", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["objects"]["counts"] == {"tp": 0, "fp": 1, "fn": 1}


def test_matching_is_one_to_one(corpus, tmp_path):
    a = corpus.clip()
    a.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    a.pred(0, "e1", "object", "chair", (0, 4, 0, 4))
    a.pred(0, "e2", "object", "chair", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["objects"]["counts"] == {"tp": 1, "fp": 1, "fn": 0}


def test_aliases_map_model_labels_onto_annotation_classes(corpus, tmp_path):
    a = corpus.clip()
    a.aliases = {"couch": "sofa"}
    a.gt_instance(0, "s", "object", "sofa", (0, 4, 0, 4))
    a.pred(0, "e", "object", "couch", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["objects"]["counts"] == {"tp": 1, "fp": 0, "fn": 0}


def test_occluded_and_ignored_regions_are_neither_detections_nor_misses(corpus, tmp_path):
    a = corpus.clip()
    a.gt_instance(0, "hidden", "object", "chair", (0, 4, 0, 4), visibility="occluded")
    a.gt_instance(0, "gone", "object", "chair", None, visibility="absent")
    a.gt_instance(0, "far", "object", "chair", (0, 2, 6, 8), visibility="occluded")
    a.gt_ignore(0, (6, 10, 6, 10))
    a.pred(0, "on-hidden", "object", "chair", (0, 4, 0, 4))  # matches an occluded instance
    a.pred(0, "in-ignore", "object", "table", (7, 9, 7, 9))  # entirely inside an ignored region
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["objects"]["counts"] == {"tp": 0, "fp": 0, "fn": 0}


def test_annotated_frames_the_scan_skipped_count_as_misses(corpus, tmp_path):
    a = corpus.clip(frames=[0, 1])
    a.gt_instance(0, "c0", "object", "chair", (0, 4, 0, 4))
    a.gt_instance(1, "c1", "object", "chair", (0, 4, 0, 4))
    a.pred(0, "e", "object", "chair", (0, 4, 0, 4))
    a.scanned = [0]
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["objects"]["counts"] == {"tp": 1, "fp": 0, "fn": 1}
    assert doc["inputs"]["clips"]["a"]["frames_unscanned"] == [1]


def test_rare_subtypes_need_their_own_recall(corpus, tmp_path):
    a = corpus.clip()
    for i in range(10):
        a.gt_instance(0, f"m{i}", "object", "mug", _px(i))
        if i < 4:  # recall for mugs is 0.4, below the 0.50 subtype floor
            a.pred(0, f"p{i}", "object", "mug", _px(i))
    for i in range(10):
        a.gt_instance(0, f"c{i}", "object", "chair", _px(i, row=2))
        a.pred(0, f"q{i}", "object", "chair", _px(i, row=2))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["objects"]
    assert gate["recall"] == pytest.approx(14 / 20)  # overall 0.70: would pass alone
    assert gate["per_class"]["mug"]["recall"] == pytest.approx(0.4)
    assert gate["pass"] is False
    assert any("mug" in r for r in gate["reasons"])


def test_vehicles_are_scored_separately_from_objects(corpus, tmp_path):
    a = corpus.clip()
    a.gt_instance(0, "car", "vehicle", "car", (0, 4, 0, 4))
    a.pred(0, "e", "vehicle", "car", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["vehicles"]["counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert doc["gates"]["objects"]["counts"] == {"tp": 0, "fp": 0, "fn": 0}
    assert doc["gates"]["objects"]["pass"] is False, "no annotated objects is not a pass"


# --- structural surfaces -----------------------------------------------------------------


def _structural_corpus(corpus):
    a = corpus.clip(frames=[0, 1, 2])
    a.gt_surface(0, "floor", (5, 10, 0, 10))  # 50 px
    a.pred_surface(0, "floor", (6, 10, 0, 10))  # 40 px, all inside: inter 40, union 50
    a.gt_surface(1, "floor", (5, 10, 0, 10))
    a.pred_surface(1, "floor", (7, 10, 0, 10))  # 30 px: inter 30, union 50
    a.gt_negative(2, "floor")
    return a


def test_floor_iou_is_pooled_over_the_union_of_pixels(corpus, tmp_path):
    _structural_corpus(corpus)
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["floors"]
    assert gate["iou"] == pytest.approx(70 / 100)
    assert gate["per_clip"]["a"] == pytest.approx(0.7)
    assert gate["negatives"] == {"frames": 1, "false_positive_frames": 0, "rate": 0.0}
    assert gate["pass"] is True


def test_ignored_pixels_leave_both_sides_of_the_iou(corpus, tmp_path):
    a = corpus.clip()
    a.gt_surface(0, "floor", (5, 10, 0, 10))
    a.pred_surface(0, "floor", (5, 10, 0, 10))
    a.gt_ignore(0, (9, 10, 0, 10))
    a.gt_negative(0, "wall")  # not under test here
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["floors"]["iou"] == pytest.approx(1.0)


def test_a_missing_ceiling_fails_even_when_everything_else_is_strong(corpus, tmp_path):
    a = corpus.clip(frames=[0, 1])
    for cls, rect in (("floor", (5, 10, 0, 10)), ("wall", (0, 5, 0, 3))):
        a.gt_surface(0, cls, rect)
        a.pred_surface(0, cls, rect)
    a.gt_surface(0, "ceiling", (0, 2, 3, 10))  # annotated, never predicted
    for i in range(10):
        a.gt_instance(0, f"o{i}", "object", "chair", _px(i, row=3))
        a.pred(0, f"p{i}", "object", "chair", _px(i, row=3))
        a.gt_instance(0, f"v{i}", "vehicle", "car", _px(i, row=4))
        a.pred(0, f"w{i}", "vehicle", "car", _px(i, row=4))
    for cls in ("floor", "wall"):
        a.gt_negative(1, cls)
    corpus.write()

    rc, doc = _score(corpus, tmp_path)

    assert doc["gates"]["floors"]["iou"] == 1.0
    assert doc["gates"]["objects"]["pass"]
    assert doc["gates"]["vehicles"]["pass"]
    assert doc["gates"]["ceilings"]["iou"] == 0.0
    assert doc["gates"]["ceilings"]["pass"] is False
    assert doc["gates"]["structural_surfaces"]["pass"] is False
    assert rc == 1


@pytest.mark.parametrize(
    ("rect", "false_positive"),
    [((0, 1, 0, 10), True), ((0, 1, 0, 1), False)],  # 10% and 1% of the image
)
def test_negative_frames_tolerate_at_most_one_percent_of_the_image(
    corpus, tmp_path, rect, false_positive
):
    a = corpus.clip()
    a.gt_negative(0, "ceiling")
    a.pred_surface(0, "ceiling", rect)
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["ceilings"]["negatives"]["false_positive_frames"] == int(false_positive)


def test_a_class_with_no_negative_frames_cannot_pass(corpus, tmp_path):
    a = corpus.clip()
    a.gt_surface(0, "wall", (0, 5, 0, 5))
    a.pred_surface(0, "wall", (0, 5, 0, 5))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["walls"]
    assert gate["iou"] == 1.0
    assert gate["negatives"]["rate"] is None
    assert gate["pass"] is False
    assert any("negative" in r for r in gate["reasons"])


def test_each_positive_clip_must_reach_the_per_clip_floor(corpus, tmp_path):
    good = corpus.clip("good", frames=[0, 1])
    good.gt_surface(0, "wall", FULL)
    good.pred_surface(0, "wall", FULL)  # IoU 1.0
    good.gt_negative(1, "wall")
    bad = corpus.clip("bad", frames=[0])
    bad.gt_surface(0, "wall", FULL)
    bad.pred_surface(0, "wall", (0, 4, 0, 10))  # 40/100 = 0.4 < 0.50
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["walls"]
    assert gate["iou"] == pytest.approx(140 / 200)  # 0.70 aggregate clears 0.65
    assert gate["per_clip"] == {"good": pytest.approx(1.0), "bad": pytest.approx(0.4)}
    assert gate["pass"] is False
    assert any("bad" in r for r in gate["reasons"])


# --- identities ------------------------------------------------------------------------------


def _track(clip, frames, car_rect=(0, 4, 0, 4)):
    for f in frames:
        clip.gt_instance(f, "car-1", "vehicle", "car", car_rect)


def test_idf1_counts_identity_switches(corpus, tmp_path):
    a = corpus.clip(frames=range(4))
    _track(a, range(4))
    a.interval(0, 3)
    for f in (0, 1):
        a.pred(f, "A", "vehicle", "car", (0, 4, 0, 4))
    for f in (2, 3):
        a.pred(f, "B", "vehicle", "car", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["identity_vehicles"]
    assert gate["idtp"] == 2
    assert gate["idf1"] == pytest.approx(2 * 2 / (4 + 4))  # 0.5
    assert gate["switches"] == 1
    assert gate["fragmentation"] == 0
    assert gate["misses"] == 0
    assert gate["pass"] is False


def test_idf1_counts_gaps_as_fragmentation_and_misses(corpus, tmp_path):
    a = corpus.clip(frames=range(4))
    _track(a, range(4))
    a.interval(0, 3)
    for f in (0, 1, 3):
        a.pred(f, "A", "vehicle", "car", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["identity_vehicles"]
    assert gate["idtp"] == 3
    assert gate["idf1"] == pytest.approx(2 * 3 / (4 + 3))
    assert gate["switches"] == 0
    assert gate["fragmentation"] == 1
    assert gate["misses"] == 1


def test_a_steady_identity_passes_and_other_family_has_no_data(corpus, tmp_path):
    a = corpus.clip(frames=range(4))
    _track(a, range(4))
    a.interval(0, 3)
    for f in range(4):
        a.pred(f, "A", "vehicle", "car", (0, 4, 0, 4))
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["identity_vehicles"]["idf1"] == 1.0
    assert doc["gates"]["identity_vehicles"]["pass"] is True
    assert doc["gates"]["identity_objects"]["pass"] is False
    assert doc["gates"]["identity_objects"]["idf1"] is None


def test_tracking_intervals_may_not_cross_a_cut(corpus, tmp_path, capsys):
    a = corpus.clip(frames=range(6), shots=[(0, 2), (3, 5)])
    a.interval(1, 4)
    corpus.write()

    with pytest.raises(SystemExit) as exc:
        _score(corpus, tmp_path)

    assert exc.value.code == 2
    assert "crosses a shot boundary" in capsys.readouterr().err


# --- registration and scale ------------------------------------------------------------------


def _registration_clip(corpus, name, errors_px, eligible=range(5)):
    """Control point at (5, 5) every frame; ``errors_px[f]`` is the predicted x offset."""
    a = corpus.clip(name, frames=list(eligible), tags=["translating"])
    a.gt_geometry(
        eligible,
        control_points=[{"frame_id": f, "id": "corner", "xy": [5.0, 5.0]} for f in eligible],
    )
    frames = [
        {
            "frame_id": f,
            "registered": True,
            "control_points": [{"id": "corner", "xy": [5.0 + e, 5.0]}],
        }
        for f, e in errors_px.items()
    ]
    for f in eligible:
        if f not in errors_px:
            frames.append({"frame_id": f, "registered": False, "control_points": []})
    a.pred_registration("registered_relative", frames)
    return a


def test_registration_reports_coverage_and_reprojection_error(corpus, tmp_path):
    _registration_clip(corpus, "a", {0: 0.0, 1: 0.02, 2: 0.04, 3: 0.05})  # frame 4 abstains
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["registration"]
    diag = math.hypot(10, 10)
    assert gate["eligible_frames"] == 5
    assert gate["registered_frames"] == 4
    assert gate["coverage"] == pytest.approx(0.8)
    # percentages of the image diagonal: 0, .02/d, .04/d, .05/d
    assert gate["median_error_pct"] == pytest.approx((0.02 + 0.04) / 2 / diag * 100)
    assert gate["p95_error_pct"] == pytest.approx(0.05 / diag * 100)  # nearest rank, n=4
    assert gate["pass"] is True


def test_unavailable_geometry_cannot_claim_registered_evaluation_frames(corpus, tmp_path, capsys):
    a = corpus.clip("unavailable", frames=range(5), tags=["translating"])
    a.gt_geometry(
        range(5),
        control_points=[{"frame_id": f, "id": "corner", "xy": [5.0, 5.0]} for f in range(5)],
    )
    a.pred_registration(
        "unavailable",
        [
            {
                "frame_id": f,
                "registered": True,
                "control_points": [{"id": "corner", "xy": [5.0, 5.0]}],
            }
            for f in range(5)
        ],
    )
    corpus.write()

    out = tmp_path / "score.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(corpus.predictions),
                "--annotations",
                str(corpus.annotations),
                "--out",
                str(out),
            ]
        )

    assert exc.value.code == 2
    assert (
        "registered: true conflicts with geometry.status 'unavailable'" in capsys.readouterr().err
    )
    assert not out.exists()


def test_registration_evaluation_missing_frame_id_is_invalid_before_report_write(
    corpus, tmp_path, capsys
):
    a = corpus.clip("malformed-registration")
    a.gt_geometry([], control_points=[])
    a.pred_registration("unavailable", [{"registered": True, "control_points": []}])
    corpus.write()
    out = tmp_path / "score.json"

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(corpus.predictions),
                "--annotations",
                str(corpus.annotations),
                "--out",
                str(out),
            ]
        )

    assert exc.value.code == 2
    assert "geometry.evaluation.frames[0].frame_id" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["fit-point", "missing-dimension-units"])
def test_registration_evaluation_must_name_independent_checks_and_units(
    corpus, tmp_path, capsys, mutation
):
    a = corpus.clip("malformed-registration")
    a.pred_registration(
        "registered_metric",
        frames=[
            {
                "frame_id": 0,
                "registered": True,
                "control_points": [{"id": "corner", "xy": [5.0, 5.0]}],
            }
        ],
        dimensions=[{"id": "door-height", "meters": 2.0}],
    )
    corpus.write()
    prediction = corpus.predictions / "malformed-registration.json"
    doc = json.loads(prediction.read_text())
    if mutation == "fit-point":
        doc["geometry"]["evaluation"]["frames"][0]["control_points"][0]["id"] = "fixture-fit"
    else:
        doc["geometry"]["evaluation"]["dimensions"][0].pop("units")
    prediction.write_text(json.dumps(doc))
    out = tmp_path / "score.json"

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(corpus.predictions),
                "--annotations",
                str(corpus.annotations),
                "--out",
                str(out),
            ]
        )

    assert exc.value.code == 2
    error = capsys.readouterr().err
    if mutation == "fit-point":
        assert "independent registration check anchors" in error
    else:
        assert "geometry.evaluation.dimensions[0]: missing 'units'" in error
    assert not out.exists()


@pytest.mark.parametrize("collision", ["annotation", "prediction", "asset", "hardlink", "symlink"])
def test_score_output_cannot_overwrite_or_alias_inputs(corpus, tmp_path, capsys, collision):
    a = corpus.clip("collision", frames=[0])
    a.gt_surface(0, "wall", (0, 1, 0, 1))
    a.pred_surface(0, "wall", (0, 1, 0, 1))
    corpus.write()
    annotation = corpus.annotations / "collision.json"
    prediction = corpus.predictions / "collision.json"
    asset = next((corpus.predictions / "collision.assets").rglob("*.png"))
    source = {
        "annotation": annotation,
        "prediction": prediction,
        "asset": asset,
    }
    if collision in source:
        out = source[collision]
    elif collision == "hardlink":
        out = tmp_path / "hardlink.json"
        out.hardlink_to(annotation)
    else:
        out = tmp_path / "symlink.json"
        out.symlink_to(annotation)
    protected = {path: path.read_bytes() for path in (annotation, prediction, asset)}

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(corpus.predictions),
                "--annotations",
                str(corpus.annotations),
                "--out",
                str(out),
            ]
        )

    assert exc.value.code == 2
    assert "--out" in capsys.readouterr().err
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_large_reprojection_errors_fail_registration(corpus, tmp_path):
    _registration_clip(corpus, "a", {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5})
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["registration"]
    assert gate["coverage"] == 1.0
    assert gate["median_error_pct"] == pytest.approx(0.3 / math.hypot(10, 10) * 100)  # ~2.1%
    assert gate["pass"] is False


def test_always_abstaining_geometry_fails_registration(corpus, tmp_path):
    _registration_clip(corpus, "a", {})
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["registration"]
    assert gate["coverage"] == 0.0
    assert gate["pass"] is False


def test_no_eligible_frames_cannot_pass_registration(corpus, tmp_path):
    corpus.clip()
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["registration"]["pass"] is False
    assert doc["gates"]["registration"]["coverage"] is None


@pytest.mark.parametrize(
    ("predicted", "status", "passes"),
    [
        (2.15, "registered_metric", True),
        (2.3, "registered_metric", False),
        (2.0, "registered_relative", False),
    ],
)
def test_metric_scale_is_checked_on_withheld_dimensions_only(
    corpus, tmp_path, predicted, status, passes
):
    a = corpus.clip(tags=["translating"])
    dims = [
        {"id": "door-height", "meters": 2.0, "uncertainty_m": 0.01, "withheld": True},
        {"id": "fit-width", "meters": 1.0, "uncertainty_m": 0.01, "withheld": False},
    ]
    a.gt_geometry([0], dimensions=dims)
    a.pred_registration(
        status,
        dimensions=[{"id": "door-height", "meters": predicted}, {"id": "fit-width", "meters": 5.0}],
    )
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["metric_scale"]
    assert [d["id"] for d in gate["dimensions"]] == ["door-height"]
    assert gate["dimensions"][0]["relative_error"] == pytest.approx(abs(predicted - 2.0) / 2.0)
    assert gate["dimensions"][0]["uncertainty_m"] == 0.01
    assert gate["pass"] is passes


def test_underconstrained_clips_must_not_claim_registration(corpus, tmp_path):
    honest = corpus.clip("honest", tags=["underconstrained"])
    honest.pred_registration("relative")
    liar = corpus.clip("liar", tags=["underconstrained"])
    liar.pred_registration("registered_metric")
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    gate = doc["gates"]["underconstrained"]
    assert gate["pass"] is False
    assert gate["clips"] == {"honest": True, "liar": False}


def test_relative_camera_frame_is_honest_for_underconstrained_clips(corpus, tmp_path):
    clip = corpus.clip("relative-camera", tags=["underconstrained"])
    depth_asset = "geometry/depth/depth-00000000.npy"
    depth_path = corpus.predictions / "relative-camera.assets" / depth_asset
    depth_path.parent.mkdir(parents=True)
    depth_path.write_bytes(b"relative test depth")
    clip.pred_geometry = {
        "status": "relative-camera-frame",
        "reason": "test relative camera frame",
        "units": "relative_depth",
        "coordinate_convention": "camera x right, y down, z forward; world-to-camera extrinsics",
        "scale_provenance": {"kind": "relative_model_prediction", "metric": False},
        "static_fusion_entities": [],
        "excluded_dynamic_entities": [],
        "frames": [
            {
                "frame_id": 0,
                "depth_asset": depth_asset,
                "intrinsics": [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]],
                "extrinsics_w2c": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                "depth_pixel_space": "source_frame_pixels",
                "preprocessing": {
                    "process_res": 14,
                    "process_res_method": "upper_bound_resize",
                    "source_size": [10, 10],
                    "processed_size": [10, 10],
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
                    "depth_resampling": "bilinear to source_size",
                    "intrinsics_pixel_space": "undistorted_source_frame_pixels",
                    "crop": None,
                    "lens_transform": {"model": "none", "applied": False},
                },
            }
        ],
    }
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["gates"]["underconstrained"]["clips"] == {"relative-camera": True}
    assert doc["gates"]["underconstrained"]["pass"] is True


# --- corpus coverage, splits and inputs -------------------------------------------------------


def test_a_small_corpus_fails_coverage_with_named_reasons(corpus, tmp_path):
    corpus.clip()
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    coverage = doc["gates"]["corpus_coverage"]
    assert coverage["pass"] is False
    failed = {c["name"] for c in coverage["checks"] if not c["pass"]}
    assert {
        "clips",
        "ceiling positive clips",
        "ceiling negative frames",
        "object subtypes",
    } <= failed


@pytest.mark.parametrize("geometry_only_abstention", [False, True])
def test_a_full_corpus_passes_coverage(corpus, tmp_path, geometry_only_abstention):
    required = list(envscore.TARGETS["required_tags"])
    for i in range(12):
        split = "development" if i < 6 else "heldout"
        a = corpus.clip(f"c{i:02d}", split=split, frames=range(30), fps=10, tags=required)
        for cls in ("floor", "wall", "ceiling"):
            a.gt_surface(0, cls, FULL)
            a.gt_negative(1, cls)
        for k in range(6):
            a.gt_instance(2, f"o{k}", "object", f"object-{k}", _px(k))
        for k in range(3):
            a.gt_instance(2, f"v{k}", "vehicle", f"vehicle-{k}", _px(k, row=1))
        a.gt_geometry(
            range(30),
            control_points=[{"frame_id": 0, "id": "corner", "xy": [5.0, 5.0]}],
            dimensions=[{"id": "door", "meters": 2.0, "uncertainty_m": 0.01, "withheld": True}],
        )
        a.interval(0, 29)  # 2.9 s at 10 fps
        if geometry_only_abstention and i == 0:
            a.run_status = "partial"
            a.perception_status = "complete"
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    coverage = doc["gates"]["corpus_coverage"]
    assert [c["name"] for c in coverage["checks"] if not c["pass"]] == []
    assert coverage["pass"] is True


def test_the_split_option_selects_which_clips_are_scored(corpus, tmp_path):
    dev = corpus.clip("dev", split="development")
    dev.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    dev.pred(0, "e", "object", "chair", (0, 4, 0, 4))
    held = corpus.clip("held", split="heldout")
    held.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    corpus.write()

    _, held_doc = _score(corpus, tmp_path)
    _, dev_doc = _score(corpus, tmp_path, "--split", "development")
    _, all_doc = _score(corpus, tmp_path, "--split", "all")

    assert held_doc["gates"]["objects"]["counts"] == {"tp": 0, "fp": 0, "fn": 1}
    assert dev_doc["gates"]["objects"]["counts"] == {"tp": 1, "fp": 0, "fn": 0}
    assert all_doc["gates"]["objects"]["counts"] == {"tp": 1, "fp": 0, "fn": 1}


def test_a_clip_without_a_prediction_scores_as_all_misses(corpus, tmp_path):
    a = corpus.clip("lonely")
    a.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    corpus.write()
    (corpus.predictions / "lonely.json").unlink()

    _, doc = _score(corpus, tmp_path)

    assert doc["inputs"]["clips"]["lonely"]["prediction"] == "missing"
    assert doc["gates"]["objects"]["counts"] == {"tp": 0, "fp": 0, "fn": 1}
    assert doc["gates"]["corpus_coverage"]["pass"] is False


def test_incomplete_runs_are_flagged_and_fail_coverage(corpus, tmp_path):
    a = corpus.clip()
    a.run_status = "partial"
    a.perception_status = "partial"
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["inputs"]["clips"]["a"]["run_status"] == "partial"
    failed = {c["name"] for c in doc["gates"]["corpus_coverage"]["checks"] if not c["pass"]}
    assert "complete prediction runs" in failed


def test_output_has_a_version_targets_and_no_composite_score(corpus, tmp_path):
    corpus.clip()
    corpus.write()

    _, doc = _score(corpus, tmp_path)

    assert doc["schema"] == envscore.SCORE_SCHEMA
    assert doc["scorer_version"] == envscore.SCORER_VERSION
    assert doc["targets"]["object_precision"] == 0.80
    flat = json.dumps(doc)
    assert "composite" not in flat
    assert "average" not in flat
    assert isinstance(doc["all_gates_passed"], bool)


def test_scoring_never_overwrites_inputs_and_writes_atomically(corpus, tmp_path):
    corpus.clip()
    corpus.write()
    out = tmp_path / "score.json"
    out.write_text("previous")

    _score(corpus, tmp_path)

    assert out.read_text() != "previous"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


# --- annotation validation ------------------------------------------------------------------


def _edit_annotation(corpus, fn):
    path = corpus.annotations / "a.json"
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(schema="other/1"), "unsupported annotation schema"),
        (lambda d: d.update(split="train"), "split"),
        (lambda d: d.update(source_sha256="abc"), "64-character"),
        (lambda d: d.update(source_sha256="z" * 64), "hexadecimal"),
        (lambda d: d["frames"][0].update(negative_classes=["table"]), "class-negative"),
        (
            lambda d: d["frames"][0]["instances"].append(
                {"id": "x", "family": "object", "class": "c", "visibility": "visible"}
            ),
            "mask",
        ),
        (lambda d: d["frames"][0].update(frame_id="zero"), "frame_id"),
        (lambda d: d["review"].update(disagreements_resolved=False), "disagreements_resolved"),
        (lambda d: d.pop("shots"), "missing 'shots'"),
    ],
)
def test_malformed_annotations_are_rejected_with_pointed_errors(
    corpus, tmp_path, capsys, edit, message
):
    a = corpus.clip()
    a.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    corpus.write()
    _edit_annotation(corpus, edit)

    with pytest.raises(SystemExit) as exc:
        _score(corpus, tmp_path)

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "a.json" in err
    assert message in err
    assert not (tmp_path / "score.json").exists()


def test_null_annotation_surface_list_is_a_pointed_error_and_preserves_prior_score(
    corpus, tmp_path, capsys
):
    corpus.clip()
    corpus.write()
    _edit_annotation(corpus, lambda d: d["frames"][0].update(surfaces=None))
    out = tmp_path / "score.json"
    out.write_text("previous report")

    with pytest.raises(SystemExit) as exc:
        _score(corpus, tmp_path)

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "a.json" in err
    assert "frames[0].surfaces: expected a list" in err
    assert out.read_text() == "previous report"


def test_non_finite_annotation_control_point_is_rejected(corpus, tmp_path, capsys):
    _registration_clip(corpus, "a", {0: 0.0}, eligible=[0])
    corpus.write()
    path = corpus.annotations / "a.json"
    path.write_text(path.read_text().replace('"xy": [5.0, 5.0]', '"xy": [1e999, 5.0]', 1))

    with pytest.raises(SystemExit) as exc:
        _score(corpus, tmp_path)

    assert exc.value.code == 2
    assert "geometry.control_points[0].xy: expected finite numbers" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda d: d["geometry"].update(eligible_frames=[0, 0]),
            "geometry.eligible_frames: duplicate frame id 0",
        ),
        (
            lambda d: d["geometry"]["control_points"].append(
                {"frame_id": 0, "id": "corner", "xy": [6.0, 6.0]}
            ),
            "geometry.control_points[1]: duplicate id 'corner' in frame 0",
        ),
    ],
)
def test_duplicate_annotation_geometry_references_are_rejected(
    corpus, tmp_path, capsys, edit, message
):
    _registration_clip(corpus, "a", {0: 0.0}, eligible=[0])
    corpus.write()
    _edit_annotation(corpus, edit)

    with pytest.raises(SystemExit) as exc:
        _score(corpus, tmp_path)

    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_masks_of_the_wrong_size_or_outside_the_directory_are_rejected(corpus, tmp_path, capsys):
    a = corpus.clip()
    a.gt_instance(0, "c", "object", "chair", (0, 4, 0, 4))
    corpus.write()
    _edit_annotation(
        corpus, lambda d: d["frames"][0]["instances"][0].update(mask="../../etc/passwd")
    )

    with pytest.raises(SystemExit):
        _score(corpus, tmp_path)

    assert "escapes the bundle" in capsys.readouterr().err


def test_missing_directories_and_empty_corpora_are_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(tmp_path / "none"),
                "--annotations",
                str(tmp_path),
                "--out",
                str(tmp_path.parent / "s.json"),
            ]
        )
    assert exc.value.code == 2
    assert "no such directory" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "environment-score",
                "--predictions",
                str(empty),
                "--annotations",
                str(empty),
                "--out",
                str(tmp_path / "s.json"),
            ]
        )
    assert exc.value.code == 2
    assert "no annotation files" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("weights", "best"),
    [
        ([[3, 2], [3, 0]], 5),  # a greedy pick of 3 first would total only 3
        ([[1, 2, 3]], 3),
        ([[5], [4]], 5),
        ([], 0),
        ([[0, 0], [0, 0]], 0),
    ],
)
def test_identity_assignment_is_optimal(weights, best):
    assert envscore._hungarian_max(weights) == best
