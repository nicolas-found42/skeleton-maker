# SPDX-License-Identifier: MIT
"""Reject malformed source clocks, nonexistent frames and unverifiable registered depth."""

import json

import numpy as np
import pytest

from skeleton_maker import envmanifest, stage_environment
from tests.env_corpus import Corpus
from tests.test_character_environment import _registered_manifest


@pytest.mark.parametrize("frame_id", [-1, 1])
def test_processed_frame_must_exist_in_source(tmp_path, frame_id):
    corpus = Corpus(tmp_path)
    corpus.clip()
    corpus.write()
    path = corpus.predictions / "a.json"
    doc = json.loads(path.read_text())
    doc["processed_frames"][0]["frame_id"] = frame_id
    with pytest.raises(envmanifest.ManifestError, match=r"frame_id.*source"):
        envmanifest.validate_manifest(doc)


@pytest.mark.parametrize(
    ("field", "values"),
    [("rate", [True, 1]), ("rate", [10, True]), ("time", [False, 10]), ("time", [0, True])],
)
def test_rational_clock_rejects_boolean_components(tmp_path, field, values):
    corpus = Corpus(tmp_path)
    corpus.clip()
    corpus.write()
    doc = json.loads((corpus.predictions / "a.json").read_text())
    if field == "rate":
        doc["source"]["frame_rate"] = values
    else:
        doc["processed_frames"][0]["time"] = values
    with pytest.raises(envmanifest.ManifestError, match=r"frame_rate|\.time"):
        envmanifest.validate_manifest(doc)


def test_registered_depth_must_be_declared_in_hashed_assets(tmp_path):
    path = _registered_manifest(tmp_path)
    doc = json.loads(path.read_text())
    doc["assets"] = []
    path.write_text(json.dumps(doc))
    with pytest.raises(envmanifest.ManifestError, match=r"depth_asset.*not listed"):
        envmanifest.load_manifest(path)


def test_registered_stage_hides_mismatched_depth_grid(tmp_path):
    path = _registered_manifest(tmp_path)
    doc = envmanifest.load_manifest(path)
    frame = doc["geometry"]["frames"][0]
    np.save(
        envmanifest.assets_dir_for(path) / frame["depth_asset"], np.ones((5, 5)), allow_pickle=False
    )
    meta = {
        "shots": [{"display_frames": [{"source_frame": 0, "camera_to_stage": np.eye(4).tolist()}]}]
    }
    result = stage_environment.prepare(path, meta, document=doc)
    assert result["frames"]["0"]["points"] == []
    assert "size" in result["frames"]["0"]["reason"]
