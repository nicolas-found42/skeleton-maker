# SPDX-License-Identifier: MIT
"""The corpus runner builds one fixed command per clip and keeps every result, including failures."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_environment_corpus.py"
_spec = importlib.util.spec_from_file_location("run_environment_corpus", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

SEMANTIC = {"clip": "vip-a", "split": "heldout", "scope": "semantic", "frame_rate": [5, 1]}
GEOMETRY = {"clip": "tum-a", "split": "heldout", "scope": "geometry", "frame_rate": [30, 1]}
REPORT = {"parameters": {"sample_step": 60}}


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    for sub in ("clips", "annotations", "references", "build-reports"):
        (corpus / sub).mkdir(parents=True)
    for annotation in (SEMANTIC, GEOMETRY):
        (corpus / "annotations" / f"{annotation['clip']}.json").write_text(json.dumps(annotation))
    (corpus / "references" / "tum-a.geometry-reference.json").write_text("{}")
    (corpus / "build-reports" / "tum-a.build-report.json").write_text(json.dumps(REPORT))
    return corpus


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_a_semantic_clip_is_scanned_at_its_own_rate_without_a_calibration(tmp_path):
    argv = runner.command(_corpus(tmp_path), SEMANTIC, tmp_path / "out", None)

    assert _flag(argv, "--sample-fps") == "5"
    assert _flag(argv, "--device") == "mps"
    assert _flag(argv, "--geometry") == "auto"
    assert "--no-cache" in argv
    assert "--calibration" not in argv


def test_a_geometry_clip_is_scanned_on_its_anchor_grid_with_its_reference(tmp_path):
    corpus = _corpus(tmp_path)

    argv = runner.command(corpus, GEOMETRY, tmp_path / "out", REPORT)

    assert _flag(argv, "--sample-fps") == "0.5"
    assert _flag(argv, "--calibration") == str(
        corpus / "references" / "tum-a.geometry-reference.json"
    )


def test_a_geometry_clip_without_a_build_report_is_refused():
    with pytest.raises(runner.RunError, match="build report"):
        runner.sample_fps(GEOMETRY, None)


def test_every_run_is_recorded_a_failure_included_and_never_overwritten(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    out = tmp_path / "out"
    calls = []

    def fake_run(argv, stdout, stderr, check):
        calls.append(argv)
        stdout.write("  12345678  maximum resident set size\n")
        manifest = Path(_flag(argv, "--out"))
        if "vip-a" in argv[4]:  # argv: time, -l, skeleton-maker, environment, clip
            manifest.write_text("{}")
        return type("Done", (), {"returncode": 0 if manifest.exists() else 1})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main([str(corpus), str(out), "--split", "all"]) == 1  # tum-a wrote nothing

    ok = json.loads((out / "vip-a.run-record.json").read_text())
    bad = json.loads((out / "tum-a.run-record.json").read_text())
    assert ok["exit_code"] == 0
    assert ok["manifest_sha256"] is not None
    assert ok["peak_rss_bytes"] == 12345678
    assert bad["exit_code"] == 1
    assert bad["manifest_sha256"] is None
    assert runner.main([str(corpus), str(out), "--split", "all"]) == 0
    assert len(calls) == 2  # the second pass ran nothing
