# SPDX-License-Identifier: MIT
"""Opt-in: the real Grounded SAM 2 worker on a real clip. Never run in CI.

Needs the worker installed (`just environment-worker-setup`) and a real clip:

    SKELETON_MAKER_REAL_WORKER=1 SKELETON_MAKER_REAL_CLIP=in/cooking.skeleton/clip.mp4 pytest -k real_worker -s

A passing run shows the worker produces a schema-valid manifest with masks and that the
cache works. It says nothing about recognition accuracy; that is measured against human labels.
"""

import json
import os
from pathlib import Path

import pytest

from skeleton_maker import cli, envmanifest

REAL = os.environ.get("SKELETON_MAKER_REAL_WORKER") == "1"
CLIP = os.environ.get("SKELETON_MAKER_REAL_CLIP", "")

pytestmark = pytest.mark.skipif(
    not (REAL and Path(CLIP).is_file()),
    reason="set SKELETON_MAKER_REAL_WORKER=1 and SKELETON_MAKER_REAL_CLIP=<clip.mp4> to run",
)


@pytest.fixture(autouse=True)
def _use_the_installed_worker(monkeypatch):
    # tests/conftest.py hides installed workers; this opt-in test wants the real one.
    monkeypatch.delenv("SKELETON_MAKER_WORKER_HOME", raising=False)


def test_real_worker_writes_a_valid_manifest_with_masks_and_reuses_its_cache(tmp_path, capsys):
    out = tmp_path / "environment.json"
    argv = [
        "environment",
        CLIP,
        "--out",
        str(out),
        "--sample-fps",
        "0.1",
        "--geometry",
        "off",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]

    assert cli.main(argv) == 0

    manifest = envmanifest.load_manifest(out)
    assert manifest["run"]["status"] == "complete"
    assert manifest["backend"]["name"] == "grounded-sam2-da3"
    assert {c["name"].split(":")[0] for c in manifest["backend"]["checkpoints"]} == {
        "grounding-dino",
        "sam2",
    }
    assert manifest["backend"]["run_stats"]["wall_seconds"] > 0
    assert any(o["mask"] for o in manifest["observations"])
    assert (tmp_path / "environment.assets" / "raw" / "detections.json").is_file()
    capsys.readouterr()

    assert cli.main([*argv[:3], str(tmp_path / "again.json"), *argv[4:]]) == 0
    assert "cache hit" in capsys.readouterr().out
    assert (
        json.loads((tmp_path / "again.json").read_text())["observations"]
        == manifest["observations"]
    )
