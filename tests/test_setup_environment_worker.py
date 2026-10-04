# SPDX-License-Identifier: MIT
"""The worker setup script and the worker's own preflight, checked without any downloads."""

import pytest

from scripts import setup_environment_worker as setup
from skeleton_maker.workers import grounded_sam2_da3 as worker


def test_verify_accepts_the_pinned_hash_and_rejects_anything_else(tmp_path):
    good = tmp_path / "model.bin"
    good.write_bytes(b"weights")
    digest = setup.sha256_of(good)

    setup.verify(good, digest)

    with pytest.raises(SystemExit, match="pinned"):
        setup.verify(good, "0" * 64)
    with pytest.raises(SystemExit, match="missing"):
        setup.verify(tmp_path / "gone.bin", digest)


def test_dry_run_lists_every_step_and_changes_nothing(tmp_path, capsys):
    rc = setup.main(["--home", str(tmp_path / "w"), "--models", str(tmp_path / "m"), "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "uv venv --python 3.12" in out
    assert "-r " in out
    assert "dl.fbaipublicfiles.com" in out
    assert "snapshot_download" in out
    assert not (tmp_path / "w").exists()
    assert not (tmp_path / "m").exists()


def test_preflight_in_a_bare_environment_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SKELETON_MAKER_MODELS", str(tmp_path))

    result = worker.preflight()

    assert result["ok"] is False
    assert any("model file" in item for item in result["missing"])
    assert "docs/environment-worker.md" in result["message"]


def test_the_requirements_file_pins_what_preflight_checks():
    text = setup.REQUIREMENTS.read_text()
    for name, version in worker.PINNED_LIBRARIES.items():
        assert f"{name}=={version}" in text


def test_pinned_models_carry_a_hash_and_a_licence():
    for info in worker.MODELS.values():
        assert len(info["sha256"]) == 64
        assert info["license"]
