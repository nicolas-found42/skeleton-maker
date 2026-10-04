# SPDX-License-Identifier: MIT
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_coverage_floor.py"


def _pyproject(tmp_path, name, floor):
    path = tmp_path / name
    body = "[tool.coverage.report]\n"
    if floor is not None:
        body += f"fail_under = {floor}\n"
    path.write_text(body, encoding="utf-8")
    return str(path)


def _run(base, head):
    return subprocess.run(  # noqa: S603  fixed arguments, running our own script
        [sys.executable, str(SCRIPT), base, head], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(("base", "head"), [(34, 34), (34, 40), (34, "34.5")])
def test_equal_or_higher_floor_passes(tmp_path, base, head):
    result = _run(_pyproject(tmp_path, "b.toml", base), _pyproject(tmp_path, "h.toml", head))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("base", "head"), [(34, 33), (34, "33.9"), (34, None)])
def test_lower_or_removed_floor_fails(tmp_path, base, head):
    result = _run(_pyproject(tmp_path, "b.toml", base), _pyproject(tmp_path, "h.toml", head))
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_base_without_a_floor_passes(tmp_path):
    result = _run(_pyproject(tmp_path, "b.toml", None), _pyproject(tmp_path, "h.toml", 34))
    assert result.returncode == 0


def test_wrong_argument_count_is_a_usage_error():
    result = subprocess.run(  # noqa: S603  fixed arguments, running our own script
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2


def test_the_real_pyproject_has_a_readable_floor():
    spec = importlib.util.spec_from_file_location("check_coverage_floor", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    floor = module.read_floor(str(SCRIPT.parent.parent / "pyproject.toml"))
    assert floor is not None
    assert floor > 0
