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
