# SPDX-License-Identifier: MIT
"""Run heavyweight environment workers in their own Python environment.

The base package never imports torch or any model library. A worker is a script that runs
under a separate virtualenv's interpreter and speaks this protocol, so the base install
(Python 3.10-3.12) stays small and a worker's dependencies cannot break it:

* ``python worker.py preflight`` prints one JSON line: ``{"protocol": 1, "ok": true,
  "devices": [...], "supports_geometry": false, "identity": {...}}``, or ``"ok": false`` with
  ``missing`` and ``message`` when a dependency or device is unusable. It must not load models.
* ``python worker.py run REQUEST.json ASSETS_DIR RESPONSE.json`` performs the scan described in
  :mod:`skeleton_maker.environment`. Progress goes to stderr, one line per update. Exit status 0
  with a valid response file is the only success.

``max_frames`` (optional) bounds one run's sampled frames. ``identity`` is everything about the worker that changes its answers (models, checkpoint
hashes, preprocessing, settings); it keys the inference cache.
"""

import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

PROTOCOL = 1
PREFLIGHT_TIMEOUT_S = 180

WORKERS_DIR = Path(__file__).with_name("workers")
#: Workers the package knows how to find, by backend name.
KNOWN_WORKERS = {"grounded-sam2-da3": "grounded_sam2_da3.py"}


class BackendError(RuntimeError):
    """A backend failed or returned something that is not a valid response."""


class BackendUnavailable(RuntimeError):
    """The backend cannot run here: a dependency, model file or device is missing."""


def worker_home() -> Path:
    override = os.environ.get("SKELETON_MAKER_WORKER_HOME")
    if override:
        return Path(override)
    return Path(os.path.expanduser("~")) / ".cache" / "skeleton-maker" / "workers"


class SubprocessBackend:
    """A backend whose work happens in another interpreter, over the worker protocol."""

    def __init__(self, name: str, python: str, script):
        self.name = name
        self.python = str(python)
        self.script = str(script)
        self._preflight_result: dict | None = None

    def _preflight(self) -> dict:
        if self._preflight_result is None:
            try:
                proc = subprocess.run(  # noqa: S603  fixed argument list, no shell
                    [self.python, self.script, "preflight"],
                    capture_output=True,
                    text=True,
                    timeout=PREFLIGHT_TIMEOUT_S,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise BackendUnavailable(
                    f"backend {self.name!r} is unavailable: cannot run its preflight ({exc})"
                ) from exc
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
            try:
                result = json.loads(lines[-1])
            except (IndexError, json.JSONDecodeError) as exc:
                tail = (proc.stderr or "").strip().splitlines()[-3:]
                raise BackendUnavailable(
                    f"backend {self.name!r} is unavailable: preflight gave no usable answer "
                    f"(exit {proc.returncode}: {' | '.join(tail) or 'no output'})"
                ) from exc
            if result.get("protocol") != PROTOCOL:
                raise BackendUnavailable(
                    f"backend {self.name!r} speaks worker protocol {result.get('protocol')!r}, "
                    f"expected {PROTOCOL}"
                )
            if result.get("ok") is not True:
                missing = ", ".join(result.get("missing", [])) or "unspecified"
                raise BackendUnavailable(
                    f"backend {self.name!r} is unavailable: {result.get('message', 'preflight failed')} "
                    f"(missing: {missing})"
                )
            self._preflight_result = result
        return self._preflight_result

    def available_devices(self) -> list[str]:
        return list(self._preflight()["devices"])

    def supports_geometry(self) -> bool:
        return bool(self._preflight().get("supports_geometry", False))

    def cache_identity(self) -> dict:
        return {"worker": self.name, **self._preflight()["identity"]}

    def max_frames(self) -> int | None:
        limit = self._preflight().get("max_frames")
        return int(limit) if limit is not None else None

    def run(self, request: dict, assets_dir: Path) -> dict:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="skeleton-maker-worker-") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            proc = subprocess.Popen(  # noqa: S603  fixed argument list, no shell
                [
                    self.python,
                    self.script,
                    "run",
                    str(request_path),
                    str(assets_dir),
                    str(response_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            tail: deque[str] = deque(maxlen=20)
            try:
                stream = proc.stderr
                if stream is None:
                    raise BackendError("cannot read the worker's progress stream")
                for line in stream:
                    line = line.rstrip()
                    tail.append(line)
                    print(f"  worker: {line}", file=sys.stderr, flush=True)
                status = proc.wait()
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
            if status != 0:
                raise BackendError(f"worker exited with status {status}: {' | '.join(tail)}")
            if not response_path.is_file():
                raise BackendError("worker exited normally but wrote no response")
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BackendError(f"worker response is malformed: {exc}") from exc
            return response


def discover(name: str) -> SubprocessBackend | None:
    """The installed worker for ``name``, or None when its environment does not exist."""
    script = KNOWN_WORKERS.get(name)
    if script is None:
        return None
    python = worker_home() / name / ".venv" / "bin" / "python"
    if not python.exists():
        return None
    return SubprocessBackend(name, str(python), WORKERS_DIR / script)
