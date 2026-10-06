# SPDX-License-Identifier: MIT
"""The corpus hash changes when, and only when, a corpus file changes."""

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "hash_corpus.py"
_spec = importlib.util.spec_from_file_location("hash_corpus", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)


def _corpus(tmp_path):
    for part in hc.PARTS:
        (tmp_path / part).mkdir()
    (tmp_path / "clips" / "a.mp4").write_bytes(b"clip")
    (tmp_path / "annotations" / "a.json").write_text("{}")
    (tmp_path / "scratch.txt").write_text("not part of the corpus")
    return tmp_path


def test_every_corpus_file_is_hashed_and_other_files_are_not(tmp_path):
    result = hc.hash_corpus(_corpus(tmp_path))

    assert result["file_count"] == 2
    assert result["files"]["clips/a.mp4"] == hashlib.sha256(b"clip").hexdigest()
    assert "scratch.txt" not in result["files"]


def test_the_tree_hash_changes_with_any_file_and_is_stable_otherwise(tmp_path):
    corpus = _corpus(tmp_path)
    before = hc.hash_corpus(corpus)["tree_sha256"]

    assert hc.hash_corpus(corpus)["tree_sha256"] == before
    (corpus / "annotations" / "a.json").write_text('{"x": 1}')
    assert hc.hash_corpus(corpus)["tree_sha256"] != before


def test_a_directory_without_clips_is_refused(tmp_path, capsys):
    assert hc.main([str(tmp_path)]) == 2
    assert "usage" in capsys.readouterr().err
