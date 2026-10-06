# SPDX-License-Identifier: MIT
import pytest


@pytest.fixture(autouse=True)
def _no_real_environment_worker(monkeypatch, tmp_path_factory):
    """A worker installed on the developer's machine must never leak into offline tests."""
    monkeypatch.setenv("SKELETON_MAKER_WORKER_HOME", str(tmp_path_factory.mktemp("no-workers")))
