# SPDX-License-Identifier: MIT
"""A stand-in environment worker that speaks the real worker protocol (tests only).

``preflight`` prints capabilities as one JSON line. ``run REQUEST ASSETS RESPONSE`` writes a
response file. ``FAKE_WORKER_MODE`` picks a behaviour: ok, crash, malformed, no_response,
hang, missing_dep, bad_contract, partial.
"""

import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.env_fakes import FakeBackend  # noqa: E402

MODE = os.environ.get("FAKE_WORKER_MODE", "ok")
PROTOCOL = 1


def preflight() -> int:
    if MODE == "missing_dep":
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "ok": False,
                    "missing": ["torch>=2.5"],
                    "message": "torch is not installed in the worker environment",
                }
            )
        )
        return 0
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "ok": True,
                "devices": ["mps", "cpu"],
                "supports_geometry": False,
                "max_frames": int(os.environ.get("FAKE_WORKER_MAX_FRAMES", "1000")),
                "identity": {"model": "stand-in", "preprocessing": "none", "settings": {"x": 1}},
            }
        )
    )
    return 0


def run(request_path: str, assets_dir: str, response_path: str) -> int:
    request = json.loads(pathlib.Path(request_path).read_text())
    print("progress: 1/2", file=sys.stderr, flush=True)
    if MODE == "crash":
        print("Traceback: simulated worker crash", file=sys.stderr)
        return 3
    if MODE == "hang":
        time.sleep(60)
    if MODE == "no_response":
        return 0
    if MODE == "malformed":
        pathlib.Path(response_path).write_text("{not json")
        return 0
    response = FakeBackend().run(request, pathlib.Path(assets_dir))
    if MODE == "bad_contract":
        response["contract"] = "other/9"
    if MODE == "partial":
        response["status"] = "partial"
        response["reason"] = "1 sampled frame(s) could not be decoded: [15]"
    print("progress: 2/2", file=sys.stderr, flush=True)
    pathlib.Path(response_path).write_text(json.dumps(response))
    return 0


if __name__ == "__main__":
    command = sys.argv[1]
    sys.exit(preflight() if command == "preflight" else run(*sys.argv[2:5]))
