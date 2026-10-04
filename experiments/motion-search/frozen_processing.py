# SPDX-License-Identifier: MIT
"""Pinned processing from commit939d1f0 for frozen experiment reproduction only."""

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
package = types.ModuleType("motion_frozen")
package.__path__ = [str(HERE)]
sys.modules[package.__name__] = package
for name, file in [("nova77", "frozen_nova77.py"), ("stage", "frozen_stage.py")]:
    spec = importlib.util.spec_from_file_location("motion_frozen." + name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
Options = sys.modules["motion_frozen.stage"].Options
build_stage = sys.modules["motion_frozen.stage"].build_stage
load_frames = sys.modules["motion_frozen.stage"].load_frames
