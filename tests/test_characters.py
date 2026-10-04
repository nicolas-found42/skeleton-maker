# SPDX-License-Identifier: MIT
"""Character specs are data; a wrong joint name silently breaks a rig in the browser."""

import copy
import json

import pytest

from skeleton_maker import character
from skeleton_maker.character import SpecError, builtin_specs, validate_spec


def test_builtin_characters_are_valid_and_complete():
    specs = builtin_specs()
    assert {"robot", "clay", "mannequin", "neon", "blocky", "critter"} <= set(specs)
    for name, spec in specs.items():
        assert spec["name"] == name
        assert len(spec["palettes"]) >= 2, (
            "auto-cast needs more than one palette to tell people apart"
        )


def _spec():
    return copy.deepcopy(builtin_specs()["robot"])


def test_unknown_joint_is_named_in_the_error():
    s = _spec()
    s["parts"][0]["to"] = "Navel"
    with pytest.raises(SpecError, match="Navel"):
        validate_spec(s)


def test_palette_must_colour_every_material():
    s = _spec()
    del s["palettes"][1]["glow"]
    with pytest.raises(SpecError, match="palette 1"):
        validate_spec(s)


def test_limb_needs_a_size():
    s = _spec()
    s["parts"][0].pop("w")
    with pytest.raises(SpecError, match="radius"):
        validate_spec(s)


def test_paired_limb_lists_must_match():
    s = _spec()
    s["parts"].append(
        {
            "type": "limb",
            "from": ["L_Hip", "R_Hip"],
            "to": ["L_Knee", "R_Knee", "Head"],
            "r": 0.05,
            "mat": "main",
        }
    )
    with pytest.raises(SpecError, match="same length"):
        validate_spec(s)


def test_unknown_material_reference():
    s = _spec()
    s["parts"][1]["mat"] = "gold"
    with pytest.raises(SpecError, match="gold"):
        validate_spec(s)


@pytest.mark.parametrize("value", [-1, 0, float("nan"), float("inf"), "wide", True])
def test_invalid_limb_width_is_a_pointed_error(value):
    s = _spec()
    s["parts"][0]["w"] = value
    with pytest.raises(SpecError, match=r"parts\[0\].w"):
        validate_spec(s)


@pytest.mark.parametrize(
    ("field", "value"), [("materials", {"main": "wrong"}), ("palettes", [None]), ("parts", [None])]
)
def test_malformed_spec_containers_are_validation_errors(field, value):
    s = _spec()
    s[field] = value
    with pytest.raises(SpecError, match=field[:-1]):
        validate_spec(s)


def test_invalid_spec_does_not_overwrite_existing_output(tmp_path):
    from skeleton_maker.character import make_stage_html

    spec = _spec()
    spec["parts"][0]["w"] = -1
    custom = tmp_path / "invalid.json"
    custom.write_text(json.dumps(spec))
    out = tmp_path / "stage.html"
    out.write_text("existing stage")
    with pytest.raises(SpecError, match=r"parts\[0\].w"):
        make_stage_html("unused-pose.json", str(out), extra_specs=[str(custom)])
    assert out.read_text() == "existing stage"


def test_invalid_json_is_a_pointed_spec_error_before_output(tmp_path):
    from skeleton_maker.character import make_stage_html

    custom = tmp_path / "broken.json"
    custom.write_text('{"name":')
    out = tmp_path / "stage.html"
    with pytest.raises(SpecError, match=r"broken.json.*line 1"):
        make_stage_html("unused-pose.json", str(out), extra_specs=[str(custom)])
    assert not out.exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("fps", 0),
        ("fps", -30),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("scale", 0),
        ("scale", -1),
        ("scale", float("nan")),
        ("fov", 0),
        ("fov", 180),
        ("fov", float("inf")),
        ("min_conf", -0.1),
        ("min_conf", 1.1),
    ],
)
def test_invalid_viewer_options_fail_before_output(tmp_path, option, value):
    from skeleton_maker.character import make_stage_html

    out = tmp_path / "stage.html"
    with pytest.raises(SpecError, match=option):
        make_stage_html("unused-pose.json", str(out), **{option: value})
    assert not out.exists()


@pytest.mark.parametrize(
    ("kind", "key", "value"),
    [
        ("prop", "size", [1, 0, 1]),
        ("prop", "pos", [0, 1]),
        ("chain", "n", 2.5),
        ("chain", "r", [0.1, -0.1]),
        ("chain", "dir", [0, 0, 0]),
        ("chain", "tip", "absent"),
    ],
)
def test_invalid_prop_and_chain_geometry_is_named(kind, key, value):
    s = copy.deepcopy(builtin_specs()["critter"])
    part = next(p for p in s["parts"] if p["type"] == kind)
    part[key] = value
    with pytest.raises(SpecError, match=key):
        validate_spec(s)


def test_template_tokens_in_user_text_do_not_replace_template_slots():
    from skeleton_maker.stage import Stage

    title = "__SPECS__ __VIEWER__"
    page = character.build_html(Stage({"shots": []}, b""), builtin_specs(), {"title": title}, title)
    assert f"<title>{title}</title>" in page
    spec_json = page.split('id="specs">')[1].split("</script>")[0]
    assert "robot" in json.loads(spec_json)


def test_html_is_self_contained_and_escapes_titles():
    from skeleton_maker.stage import build_stage
    from tests.test_stage import clip

    stage = build_stage(clip(20))
    page = character.build_html(
        stage, builtin_specs(), {"character": "auto", "title": "</script><b>x"}, "</script><b>x"
    )
    assert "</script><b>x" not in page.split("<title>")[1].split("</title>")[0]
    assert 'src="http' not in page, "the page must not need the network"
    assert 'href="http' not in page, "the page must not need the network"
    assert page.count("<script") == page.count("</script>")
    # the options JSON must not be able to close its own script element
    opts = page.split('id="options">')[1].split("</script>")[0]
    assert json.loads(opts.replace("<\\/", "</"))["title"] == "</script><b>x"


def test_chain_requires_a_single_joint_name():
    s = copy.deepcopy(builtin_specs()["critter"])
    chain = next(p for p in s["parts"] if p["type"] == "chain")
    chain["joint"] = ["Hips", "Chest"]
    with pytest.raises(SpecError, match="single joint"):
        validate_spec(s)


def test_script_json_escapes_html_parser_control_sequences():
    from skeleton_maker.stage import Stage

    title = "<!--<script>"
    page = character.build_html(Stage({"shots": []}, b""), builtin_specs(), {"title": title}, title)
    options = page.split('id="options">')[1].split("</script>")[0]
    assert "<" not in options
    assert json.loads(options)["title"] == title


def test_video_filename_is_encoded_as_a_url(tmp_path):
    from tests.test_stage import clip

    pose = tmp_path / "pose.json"
    pose.write_text("\n".join(json.dumps({"frame_id": f, "detections": d}) for f, d in clip(20)))
    out = tmp_path / "stage.html"
    character.make_stage_html(str(pose), str(out), video=str(tmp_path / "a#b?c%d.mp4"))
    options = out.read_text().split('id="options">')[1].split("</script>")[0]
    assert json.loads(options)["video"] == "a%23b%3Fc%25d.mp4"


def test_missing_spec_is_named_before_output(tmp_path):
    out = tmp_path / "stage.html"
    with pytest.raises(SpecError, match=r"cannot read spec.*missing\.json"):
        character.make_stage_html(
            "unused.json", str(out), extra_specs=[str(tmp_path / "missing.json")]
        )
    assert not out.exists()
