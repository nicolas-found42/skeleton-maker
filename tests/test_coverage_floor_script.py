# SPDX-License-Identifier: MIT
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_coverage_floor.py"


def _toml(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _floor(tmp_path, name, value):
    body = "[tool.coverage.report]\n"
    if value is not None:
        body += f"fail_under = {value}\n"
    return _toml(tmp_path, name, body)


def _run(base, head):
    return subprocess.run(  # noqa: S603  fixed arguments, running our own script
        [sys.executable, str(SCRIPT), base, head], capture_output=True, text=True, check=False
    )


def _load_script():
    spec = importlib.util.spec_from_file_location("check_coverage_floor", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("base", "head"), [(34, 34), (34, 40), (34, "34.5"), (34.5, 35)])
def test_equal_or_higher_floor_passes(tmp_path, base, head):
    result = _run(_floor(tmp_path, "b.toml", base), _floor(tmp_path, "h.toml", head))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("base", "head"), [(34, 33), (34, "33.9"), (34, None)])
def test_lower_or_removed_floor_fails(tmp_path, base, head):
    result = _run(_floor(tmp_path, "b.toml", base), _floor(tmp_path, "h.toml", head))
    assert result.returncode == 1
    assert "error:" in result.stderr


def test_base_without_a_floor_passes(tmp_path):
    result = _run(_floor(tmp_path, "b.toml", None), _floor(tmp_path, "h.toml", 34))
    assert result.returncode == 0


def test_an_unrelated_fail_under_earlier_in_the_file_is_ignored(tmp_path):
    other = "[tool.other]\nfail_under = 0\n\n"
    base = _toml(tmp_path, "b.toml", other + "[tool.coverage.report]\nfail_under = 34\n")
    head = _toml(tmp_path, "h.toml", other + "[tool.coverage.report]\nfail_under = 30\n")
    result = _run(base, head)
    assert result.returncode == 1
    assert "from 34 to 30" in result.stderr


def test_a_fail_under_in_another_table_is_not_the_coverage_floor(tmp_path):
    base = _floor(tmp_path, "b.toml", 34)
    head = _toml(tmp_path, "h.toml", "[tool.other]\nfail_under = 99\n")
    result = _run(base, head)
    assert result.returncode == 1
    assert "removed" in result.stderr


@pytest.mark.parametrize(
    "text",
    [
        "[tool.coverage.report]\n  fail_under = 34\n",
        '[tool.coverage.report]\n"fail_under" = 34\n',
        "[tool.coverage.report]\nfail_under = 34 # keep this at or above the measured floor\n",
        "[tool.coverage]\nreport = { fail_under = 34 }\n",
    ],
)
def test_every_valid_spelling_of_the_same_floor_passes(tmp_path, text):
    result = _run(_floor(tmp_path, "b.toml", 34), _toml(tmp_path, "h.toml", text))
    assert result.returncode == 0, result.stderr


def test_a_non_numeric_floor_is_an_error(tmp_path):
    head = _toml(tmp_path, "h.toml", '[tool.coverage.report]\nfail_under = "34"\n')
    result = _run(_floor(tmp_path, "b.toml", 34), head)
    assert result.returncode == 1
    assert "must be a number" in result.stderr


def test_malformed_toml_and_missing_files_are_errors_not_passes(tmp_path):
    bad = _toml(tmp_path, "bad.toml", "[tool.coverage.report\nfail_under = 34\n")
    assert _run(_floor(tmp_path, "b.toml", 34), bad).returncode == 1
    assert _run(_floor(tmp_path, "b.toml", 34), str(tmp_path / "missing.toml")).returncode == 1


def test_wrong_argument_count_is_a_usage_error():
    result = subprocess.run(  # noqa: S603  fixed arguments, running our own script
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2


def test_the_real_pyproject_has_a_readable_floor():
    floor = _load_script().read_floor(str(ROOT / "pyproject.toml"))
    assert floor is not None
    assert floor > 0


def test_coverage_is_always_run_with_the_pyproject_config():
    # Coverage prefers .coveragerc, setup.cfg and tox.ini over pyproject.toml, which would let the
    # guard watch a floor that the tests do not use. Pin the config and keep the others absent.
    for path in (ROOT / ".github" / "workflows" / "ci.yml", ROOT / "justfile"):
        pytest_lines = [
            line for line in path.read_text(encoding="utf-8").splitlines() if "-m pytest" in line
        ]
        assert pytest_lines, path
        assert all("--cov-config=pyproject.toml" in line for line in pytest_lines), path
    for name in (".coveragerc", "setup.cfg", "tox.ini"):
        assert not (ROOT / name).exists(), f"{name} would override the coverage floor"
