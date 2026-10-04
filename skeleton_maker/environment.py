# SPDX-License-Identifier: MIT
"""``skeleton-maker environment``: scan a clip's surroundings into a versioned manifest.

This module owns option validation, the backend contract and the run's status handling.
Heavyweight perception and geometry live behind :class:`Backend`, in a separate worker
environment; the base package only knows the request/response contract below.

Backend contract (``skeleton-maker.environment-backend/1``)
----------------------------------------------------------
``Backend.run(request, assets_dir)`` receives a JSON-able request::

    {"contract": ..., "video": "/abs/clip.mp4", "device": "cpu", "geometry": "off|auto|required",
     "requested_labels": [], "poses": null, "source": {width, height, frame_rate, frame_count},
     "shots": [{"id": "shot-0", "first_frame": 0, "last_frame": 19}, ...],
     "frames": [{"frame_id": 0, "time": [0, 1]}, ...]}

and returns a response ``dict`` carrying ``contract``, ``status`` (``complete`` or
``partial`` plus a ``reason``), ``backend`` (``name``, ``version``, ``checkpoints``),
``device``, ``entities``, ``observations`` and ``geometry``. Observation masks are
files the backend wrote under ``assets_dir`` and names by relative path. Shots are decided by
the command from the video (see :mod:`skeleton_maker.shots`); entities name one of the
request's shot ids and are only observed inside it. Coordinates are full decoded
source-frame pixels. Anything that does not validate is a failed run.
"""

import os
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import NoReturn, Protocol

from . import __version__, artifacts, envmanifest, poses, shots
from .envmanifest import SCHEMA_VERSION, ManifestError
from .utils import die, probe_video

BACKEND_CONTRACT = "skeleton-maker.environment-backend/1"

#: The backend a plain ``environment`` run asks for; the worker that provides it is a
#: separate install, so it is absent from :data:`BACKENDS` in the base package.
DEFAULT_BACKEND = "grounded-sam2-da3"

GEOMETRY_MODES = ("off", "auto", "required")
MIN_SAMPLE_FPS = 0.05
MAX_SAMPLE_FPS = 30.0
#: Cap on sampled frames per run, so a long clip at a high rate cannot request unbounded work.
MAX_SAMPLED_FRAMES = 20000

EXIT_INVALID = 2
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130

#: Geometry statuses that mean scene and skeletons share one frame.
REGISTERED = ("registered_relative", "registered_metric")


class Backend(Protocol):
    name: str

    def available_devices(self) -> list[str]: ...

    def supports_geometry(self) -> bool: ...

    def run(self, request: dict, assets_dir: Path) -> dict: ...


#: Installed backends by name. Workers register themselves here; tests substitute a
#: deterministic fake with ``monkeypatch.setitem``. Nothing here is selectable by env var.
BACKENDS: dict[str, Backend] = {}


class BackendError(RuntimeError):
    """A backend failed or returned something that is not a valid response."""


def add_cli(subparsers) -> None:
    p = subparsers.add_parser(
        "environment",
        help="scan a clip's surfaces, objects and vehicles into an environment manifest",
    )
    p.add_argument("video", help="the conformed clip (the one whose frame ids any poses describe)")
    p.add_argument("--out", help="manifest to write (default: <video>.environment.json)")
    p.add_argument(
        "--poses",
        help="existing pose output (JSON Lines) for this exact clip; reused, no NIM call",
    )
    p.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"installed perception backend (default: {DEFAULT_BACKEND})",
    )
    p.add_argument(
        "--geometry",
        choices=GEOMETRY_MODES,
        default="auto",
        help="off: semantics only; auto: add geometry when the backend can, else report it "
        "unavailable; required: fail unless a valid shared registration is produced",
    )
    p.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help=f"frames per second to scan, {MIN_SAMPLE_FPS}-{MAX_SAMPLE_FPS} and at most the "
        "clip's rate (default: 2)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="auto, or a device the backend reports (for example cpu, cuda, mps)",
    )
    p.set_defaults(func=run_cli)


def _fail_options(msg: str) -> NoReturn:
    die(msg, EXIT_INVALID)


def _probe(video: str) -> dict:
    if not os.path.isfile(video):
        _fail_options(f"no such file: {video}")
    try:
        info = probe_video(video)
    except Exception as exc:
        _fail_options(f"cannot read {video} as a video: {exc}")
    if info["width"] <= 0 or info["height"] <= 0:
        _fail_options(f"{video} has no video stream")
    try:
        rate = Fraction(info["r_frame_rate"])
    except (ValueError, ZeroDivisionError):
        rate = Fraction(0)
    if rate <= 0:
        _fail_options(f"{video} has no usable frame rate")
    if abs(info["r_fps"] - info["avg_fps"]) > 0.01:
        _fail_options(
            f"{video} has a variable frame rate ({info['r_fps']:.3f} vs {info['avg_fps']:.3f}); "
            "frame ids cannot be mapped to time. Conform it with `skeleton-maker clip` first"
        )
    info["rate"] = rate
    if info["nb_frames"] is not None:
        info["frame_count"], info["frame_count_source"] = info["nb_frames"], "container"
    else:
        info["frame_count"] = round(info["duration"] * float(rate))
        info["frame_count_source"] = "duration_estimate"
    if info["frame_count"] <= 0:
        _fail_options(f"{video} contains no frames")
    return info


def _load_pose_join(path: str, info: dict, clip_sha256: str) -> tuple[dict, list[dict]]:
    """Read and validate existing poses for this clip; the NIM is never involved."""
    if not os.path.isfile(path):
        _fail_options(f"no such file: {path}")
    try:
        records = poses.read_records(path)
        association = poses.check_against_clip(
            records,
            frame_count=info["frame_count"],
            width=info["width"],
            height=info["height"],
            clip_sha256=clip_sha256,
        )
    except poses.PoseFileError as exc:
        _fail_options(str(exc))
    block = {
        "path": str(Path(path).resolve()),
        "association": association,
        "frame_count": len(records),
        **poses.summarize(records),
    }
    return block, records


def _resolve_backend(name: str) -> Backend:
    backend = BACKENDS.get(name)
    if backend is None:
        installed = ", ".join(sorted(BACKENDS)) or "none"
        _fail_options(
            f"environment backend {name!r} is not installed (installed: {installed}). "
            "Set up the backend worker described in docs/environment.md, or choose another "
            "with --backend"
        )
    return backend


def _resolve_device(backend: Backend, wanted: str) -> str:
    available = backend.available_devices()
    if not available:
        _fail_options(f"backend {backend.name!r} reports no usable device")
    if wanted == "auto":
        return available[0]
    if wanted not in available:
        _fail_options(
            f"device {wanted!r} is not available for backend {backend.name!r} "
            f"(available: {', '.join(available)})"
        )
    return wanted


def _validate_response(resp, request: dict, mode: str) -> None:
    """Reject anything that is not a well-formed backend response before it is trusted."""
    if not isinstance(resp, dict):
        raise BackendError("backend returned a non-object response")
    if resp.get("contract") != BACKEND_CONTRACT:
        raise BackendError(
            f"backend speaks contract {resp.get('contract')!r}, expected {BACKEND_CONTRACT!r}"
        )
    status = resp.get("status")
    if status not in ("complete", "partial"):
        raise BackendError(f"backend returned status {status!r}; only complete or partial")
    if status == "partial" and not isinstance(resp.get("reason"), str):
        raise BackendError("a partial backend response must explain itself in 'reason'")
    info = resp.get("backend")
    if not isinstance(info, dict) or not all(
        isinstance(info.get(k), str) for k in ("name", "version")
    ):
        raise BackendError("backend response lacks backend.name / backend.version")
    if not isinstance(resp.get("device"), str):
        raise BackendError("backend response lacks 'device'")
    src = request["source"]
    try:
        envmanifest.validate_content(
            {**resp, "shots": request["shots"]},
            width=src["width"],
            height=src["height"],
            frame_ids=[f["frame_id"] for f in request["frames"]],
        )
    except ManifestError as exc:
        raise BackendError(f"backend returned a malformed response: {exc}") from exc
    if mode == "required" and resp["geometry"]["status"] not in REGISTERED:
        raise BackendError(
            f"--geometry required, but the backend produced geometry status "
            f"{resp['geometry']['status']!r} ({resp['geometry']['reason']}); "
            "no valid shared registration"
        )


def _collect_assets(assets_dir: Path) -> list[dict]:
    entries = []
    if not assets_dir.exists():
        return entries
    for path in sorted(assets_dir.rglob("*")):
        if path.is_symlink():
            raise BackendError(f"backend wrote a symbolic link into its assets: {path.name}")
        if path.is_file():
            entries.append(
                {
                    "path": path.relative_to(assets_dir).as_posix(),
                    "sha256": artifacts.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return entries


def _geometry_block(mode: str, backend: Backend, resp: dict | None) -> dict:
    if mode == "off":
        return {"status": "not_requested", "mode": mode, "reason": "--geometry off"}
    if resp is None or not backend.supports_geometry():
        return {
            "status": "unavailable",
            "mode": mode,
            "reason": f"backend {backend.name!r} has no geometry support",
        }
    return {**resp["geometry"], "mode": mode}


def run_cli(args) -> int:
    video = args.video
    out = Path(args.out or os.path.splitext(video)[0] + ".environment.json")
    if not MIN_SAMPLE_FPS <= args.sample_fps <= MAX_SAMPLE_FPS:
        _fail_options(f"--sample-fps must be between {MIN_SAMPLE_FPS} and {MAX_SAMPLE_FPS}")
    info = _probe(video)
    if args.sample_fps > float(info["rate"]):
        _fail_options(
            f"--sample-fps {args.sample_fps} exceeds the clip's frame rate {float(info['rate']):g}"
        )
    if out.resolve() == Path(video).resolve():
        _fail_options("--out would overwrite the source video")
    source_sha256 = artifacts.sha256_file(video)
    pose_block, pose_records = (
        _load_pose_join(args.poses, info, source_sha256) if args.poses else (None, None)
    )
    backend = _resolve_backend(args.backend)
    device = _resolve_device(backend, args.device)
    if args.geometry == "required" and not backend.supports_geometry():
        _fail_options(
            f"--geometry required, but backend {backend.name!r} has no geometry support "
            "(see docs/environment.md)"
        )

    clock = artifacts.FrameClock(info["rate"].numerator, info["rate"].denominator)
    frame_ids = clock.sample(info["frame_count"], args.sample_fps)
    if len(frame_ids) > MAX_SAMPLED_FRAMES:
        _fail_options(
            f"{len(frame_ids)} frames would be scanned; the limit is {MAX_SAMPLED_FRAMES}. "
            "Lower --sample-fps or use a shorter clip"
        )
    source = {
        "path": os.path.abspath(video),
        "sha256": source_sha256,
        "width": info["width"],
        "height": info["height"],
        "frame_rate": [info["rate"].numerator, info["rate"].denominator],
        "frame_count": info["frame_count"],
        "frame_count_source": info["frame_count_source"],
        "duration_s": info["duration"],
    }
    frames = [{"frame_id": i, "time": clock.time_pair(i)} for i in frame_ids]
    print("environment: detecting shot boundaries")
    try:
        cuts = shots.detect_cuts(video, info["frame_count"], frame_ids)
    except shots.ShotDetectionError as exc:
        die(f"environment scan failed: {exc}")
    shot_list = shots.build_shots(cuts, info["frame_count"], clock)
    detection = shots.reconcile(
        cuts,
        shots.pose_stage_shots(pose_records, float(info["rate"])) if pose_records else None,
        clock,
    )
    request = {
        "contract": BACKEND_CONTRACT,
        "video": source["path"],
        "device": device,
        "geometry": args.geometry,
        "requested_labels": [],
        "poses": pose_block,
        "source": {k: source[k] for k in ("width", "height", "frame_rate", "frame_count")},
        "shots": [{k: sh[k] for k in ("id", "first_frame", "last_frame")} for sh in shot_list],
        "frames": frames,
    }
    print(
        f"environment: backend {backend.name} on {device}, {len(frames)} frames "
        f"(~{args.sample_fps:g} fps, geometry {args.geometry})"
    )

    with artifacts.staging_dir(out) as stage:
        assets_stage = stage / "assets"
        assets_stage.mkdir()
        try:
            resp = backend.run(request, assets_stage)
            _validate_response(resp, request, args.geometry)
            assets = _collect_assets(assets_stage)
        except KeyboardInterrupt:
            print("error: interrupted; no environment manifest was written", file=sys.stderr)
            return EXIT_INTERRUPTED
        except BackendError as exc:
            die(f"environment scan failed: {exc}")
        except Exception as exc:
            die(f"environment scan failed: backend {backend.name!r} raised {exc!r}")

        manifest = {
            "schema": SCHEMA_VERSION,
            "run": {
                "status": resp["status"],
                **({"reason": resp["reason"]} if resp["status"] != "complete" else {}),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool": {"name": "skeleton-maker", "version": __version__},
            },
            "source": source,
            "processed_frames": [
                {"frame_id": i, "time": clock.time_pair(i), "time_s": float(clock.time(i))}
                for i in frame_ids
            ],
            "poses": pose_block,
            "frame_range": [frame_ids[0], frame_ids[-1]],
            "shots": shot_list,
            "shot_detection": detection,
            "entities": resp["entities"],
            "observations": resp["observations"],
            "geometry": _geometry_block(args.geometry, backend, resp),
            "backend": {**resp["backend"], "device": resp["device"]},
            "config": {
                "requested_labels": [],
                "geometry_mode": args.geometry,
                "sample_fps": args.sample_fps,
                "device": device,
            },
            "assets": assets,
        }
        try:
            envmanifest.validate_manifest(manifest)
            envmanifest.check_assets(manifest, assets_stage)
        except ManifestError as exc:
            die(f"environment scan failed: backend output is invalid: {exc}")
        text = envmanifest.dumps(manifest)
        artifacts.publish(
            out, text, assets_stage if assets else None, envmanifest.assets_dir_for(out)
        )

    _print_summary(manifest, out)
    return EXIT_PARTIAL if manifest["run"]["status"] == "partial" else 0


def _print_summary(manifest: dict, out: Path) -> None:
    status = manifest["run"]["status"]
    families: dict[str, int] = {}
    for ent in manifest["entities"]:
        families[ent["family"]] = families.get(ent["family"], 0) + 1
    found = ", ".join(f"{k} {v}" for k, v in sorted(families.items())) or "none"
    first, last = manifest["frame_range"]
    print(f"environment: {status}")
    print(f"  entities : {found}")
    print(f"  frames {first}-{last} ({len(manifest['processed_frames'])} scanned)")
    geometry = manifest["geometry"]
    print(f"  geometry : {geometry['status']} ({geometry['reason']})")
    extra = f" (+{len(manifest['assets'])} assets in {envmanifest.assets_dir_for(out)})"
    print(f"  wrote {out}{extra if manifest['assets'] else ''}")
    if status != "complete":
        print(
            f"warning: this scan is {status}: {manifest['run']['reason']}. "
            f"Exiting {EXIT_PARTIAL} so scripts do not mistake it for a complete run.",
            file=sys.stderr,
        )
