# SPDX-License-Identifier: MIT
"""Exercise the distribution outside the editable source tree, without network access."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_wheel_can_list_characters_and_assemble_html(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
        shutil.copy(root / name, source / name)
    shutil.copytree(
        root / "skeleton_maker",
        source / "skeleton_maker",
        ignore=shutil.ignore_patterns("_gen", "__pycache__"),
    )
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    subprocess.run(  # noqa: S603 -- fixed interpreter and trusted build script in a temporary copy
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel(" + repr(str(wheels)) + ")",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = tmp_path / "installed"
    with ZipFile(next(wheels.glob("*.whl"))) as wheel:
        wheel.extractall(installed)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(installed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
from skeleton_maker.character import builtin_specs, build_html
from skeleton_maker.stage import Stage
assert Path(__import__('skeleton_maker').__file__).resolve().is_relative_to(Path.cwd())
specs = builtin_specs()
assert set(specs) == {'robot', 'clay', 'mannequin', 'neon', 'blocky', 'critter'}
page = build_html(Stage({'shots': []}, b''), specs, {}, 'wheel smoke')
assert 'class Rig' in page and 'three-src' in page and 'wheel smoke' in page
print('six characters and self-contained HTML available from wheel')
""",
        ],
        cwd=installed,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "six characters" in result.stdout
