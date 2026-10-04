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

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import NoReturn, Protocol

import cv2
import numpy as np

from . import (
    __version__,
    artifacts,
    envgeometry,
    envmanifest,
    envregistration,
    envworkers,
    labels,
    poses,
    shots,
)
from .envmanifest import SCHEMA_VERSION, ManifestError
from .envworkers import BackendError, BackendUnavailable
from .path_safety import is_same_or_ancestor, paths_overlap, same_path
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

CACHE_SCHEMA = "skeleton-maker.environment-cache/1"

EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130

#: Geometry statuses that mean scene and skeletons share one frame.
REGISTERED = ("registered_relative", "registered_metric")


class Backend(Protocol):
    name: str

    def available_devices(self) -> list[str]: ...

    def supports_geometry(self) -> bool: ...

    def max_frames(self) -> int | None:
        """Most sampled frames one run may request, or None when unbounded."""
        ...

    def cache_identity(self) -> dict | None:
        """Everything that changes this backend's answers, or None to opt out of caching."""
        ...

    def run(self, request: dict, assets_dir: Path) -> dict: ...


#: Installed backends by name. Workers register themselves here; tests substitute a
#: deterministic fake with ``monkeypatch.setitem``. Nothing here is selectable by env var.
BACKENDS: dict[str, Backend] = {}


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "skeleton-maker" / "environment"


def cache_key(identity: dict, request: dict) -> str:
    """SHA-256 over everything that can change a backend's answer.

    That is the input (source hash, scanned frames, shots, any poses), the labels, the
    geometry mode and device, and the backend's own identity: model, checkpoint hashes,
    preprocessing and settings. The video's path is deliberately not part of it.
    """
    poses_block = request.get("poses")
    payload = {
        "contract": BACKEND_CONTRACT,
        "identity": identity,
        "source_sha256": request["source_sha256"],
        "frames": [f["frame_id"] for f in request["frames"]],
        "shots": request["shots"],
        "poses": None
        if poses_block is None
        else {k: v for k, v in poses_block.items() if k != "path"},
        "requested_labels": request["requested_labels"],
        "label_vocabulary": request.get("label_vocabulary"),
        "geometry": request["geometry"],
        "device": request["device"],
        "geometry_reference": request.get("geometry_reference"),
        "geometry_device": request.get("geometry_device"),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _cache_get(root: Path, key: str, assets_stage: Path) -> dict | None:
    """The cached response for ``key``, with its assets copied into ``assets_stage``."""
    entry_dir = root / key
    path = entry_dir / "entry.json"
    if not path.is_file():
        return None
    try:
        entry = envmanifest.loads(path.read_text(encoding="utf-8"))
        if entry.get("schema") != CACHE_SCHEMA or entry.get("key") != key:
            raise ManifestError("wrong schema or key")
        for asset in entry["assets"]:
            target = envmanifest.resolve_asset(entry_dir / "assets", asset["path"], "cache asset")
            if artifacts.sha256_file(target) != asset["sha256"]:
                raise ManifestError(f"cache asset {asset['path']!r} changed")
        shutil.copytree(entry_dir / "assets", assets_stage, dirs_exist_ok=True)
        if not isinstance(entry["response"], dict):
            raise ManifestError("no response")
        return entry["response"]
    except (OSError, KeyError, TypeError, ManifestError):
        print("environment: ignoring unreadable cache entry")
        shutil.rmtree(assets_stage, ignore_errors=True)
        assets_stage.mkdir()
        return None


def _cache_put(root: Path, key: str, identity: dict, resp: dict, assets_stage: Path, assets: list):
    """Store a validated complete response; best effort, a cache failure never fails the scan."""
    tmp = root / f".tmp-{os.getpid()}-{key[:12]}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        shutil.copytree(assets_stage, tmp / "assets")
        entry = {
            "schema": CACHE_SCHEMA,
            "key": key,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity": identity,
            "response": resp,
            "assets": assets,
        }
        (tmp / "entry.json").write_text(json.dumps(entry, allow_nan=False), encoding="utf-8")
        shutil.rmtree(root / key, ignore_errors=True)
        os.replace(tmp, root / key)
    except OSError as exc:
        print(f"warning: could not write the inference cache: {exc}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
        "--classes",
        help='extra labels to look for, comma separated, each optionally "label=family" '
        "(surface, object or vehicle; default object), added to the default preset",
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
    p.add_argument(
        "--calibration",
        help="versioned camera calibration and optional measured scene anchors (JSON)",
    )
    p.add_argument(
        "--cache-dir",
        help="inference cache (default: ~/.cache/skeleton-maker/environment). Reused only when "
        "the clip, frames, labels, settings, device and backend identity all match",
    )
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    p.add_argument(
        "--viewer",
        metavar="HTML",
        help="export an offline video, skeleton and environment viewer with local assets",
    )
    p.add_argument(
        "--overlay",
        metavar="MP4",
        help="render environment observations and any supplied poses onto the source video",
    )
    p.set_defaults(func=run_cli)


def _fail_options(msg: str) -> NoReturn:
    die(msg, EXIT_INVALID)


def validate_artifact_paths(*, inputs, files, directories) -> None:
    """Reject output collisions and nesting that could replace protected inputs."""
    protected = [(label, Path(path).resolve()) for label, path in inputs if path is not None]
    file_outputs = [(label, Path(path).resolve()) for label, path in files if path is not None]
    directory_outputs = [
        (label, Path(path).resolve()) for label, path in directories if path is not None
    ]

    for label, output in file_outputs:
        for input_label, source in protected:
            if same_path(output, source):
                raise ManifestError(f"{label} would overwrite the {input_label}: {source}")
    for label, output_dir in directory_outputs:
        for input_label, source in protected:
            if is_same_or_ancestor(output_dir, source):
                raise ManifestError(f"{label} would contain or replace the {input_label}: {source}")

    for index, (label, output) in enumerate(file_outputs):
        for other_label, other in file_outputs[index + 1 :]:
            if paths_overlap(output, other):
                raise ManifestError(f"{label} conflicts with {other_label}: {output} / {other}")
        for other_label, other_dir in directory_outputs:
            if paths_overlap(output, other_dir):
                raise ManifestError(f"{label} conflicts with {other_label}: {output} / {other_dir}")
    for index, (label, output_dir) in enumerate(directory_outputs):
        for other_label, other_dir in directory_outputs[index + 1 :]:
            if paths_overlap(output_dir, other_dir):
                raise ManifestError(
                    f"{label} conflicts with {other_label}: {output_dir} / {other_dir}"
                )


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
        "file_sha256": artifacts.sha256_file(path),
        **poses.summarize(records),
    }
    return block, records


def _resolve_backend(name: str) -> Backend:
    backend = BACKENDS.get(name) or envworkers.discover(name)
    if backend is None:
        installed = ", ".join(sorted(BACKENDS)) or "none"
        _fail_options(
            f"environment backend {name!r} is not installed (installed: {installed}). "
            "Set up the worker described in docs/environment-worker.md, or choose another "
            "with --backend"
        )
    return backend


def _resolve_device(backend: Backend, wanted: str) -> str:
    try:
        available = backend.available_devices()
    except BackendUnavailable as exc:
        _fail_options(str(exc))
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


def preflight_options(args) -> tuple[list[dict], Backend, str]:
    """Validate environment settings and resolve a backend before pipeline inference."""
    if not MIN_SAMPLE_FPS <= args.sample_fps <= MAX_SAMPLE_FPS:
        _fail_options(f"--sample-fps must be between {MIN_SAMPLE_FPS} and {MAX_SAMPLE_FPS}")
    try:
        vocabulary = labels.build_vocabulary(labels.parse_classes(args.classes))
    except ValueError as exc:
        _fail_options(str(exc))
    backend = _resolve_backend(args.backend)
    device = _resolve_device(backend, args.device)
    if args.geometry == "required" and not backend.supports_geometry():
        worker = envworkers.discover_geometry()
        if worker is None:
            _fail_options(
                "--geometry required needs an independent DA3 worker; install it with "
                "`just da3-geometry-setup`"
            )
        try:
            available = worker.available_devices()
            if not available:
                _fail_options("--geometry required, but DA3 reports no usable device")
            if args.device != "auto" and args.device not in available:
                _fail_options(
                    f"device {args.device!r} is not available for DA3 geometry "
                    f"(available: {', '.join(available)})"
                )
        except BackendUnavailable as exc:
            _fail_options(f"--geometry required: {exc}")
    return vocabulary, backend, device


def validate_required_reference(reference: dict | None, backend: Backend) -> None:
    """Fail before hosted inference when required DA3 registration lacks fit/check inputs."""
    if backend.supports_geometry():
        return
    if reference is None:
        _fail_options(
            "--geometry required needs a right-handed metric world_frame, at least three "
            "identified fit anchors and an independent identified check anchor with "
            "uncertainty_m"
        )
    fit = [anchor for anchor in reference.get("anchors", []) if anchor.get("role") == "fit"]
    check = [anchor for anchor in reference.get("anchors", []) if anchor.get("role") == "check"]
    if (
        not reference.get("world_frame")
        or len(fit) < 3
        or not check
        or any(not anchor.get("id") or not anchor.get("uncertainty_m") for anchor in fit + check)
    ):
        _fail_options(
            "--geometry required needs a right-handed metric world_frame, at least three "
            "identified fit anchors and an independent identified check anchor with "
            "uncertainty_m"
        )


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
            vocabulary={v["label"]: v["family"] for v in request["label_vocabulary"]},
            width=src["width"],
            height=src["height"],
            frame_ids=[f["frame_id"] for f in request["frames"]],
        )
    except ManifestError as exc:
        raise BackendError(f"backend returned a malformed response: {exc}") from exc
    if mode == "required" and resp["geometry"]["status"] != "registered_metric":
        raise BackendError(
            f"--geometry required, but the backend produced geometry status "
            f"{resp['geometry']['status']!r} ({resp['geometry']['reason']}); "
            "no valid shared metric registration"
        )


def _check_masks(resp: dict, request: dict, assets_dir: Path) -> None:
    """Masks must show visible extent only: readable, source-sized, non-empty, inside their box."""
    src = request["source"]
    shape = (src["height"], src["width"])
    for obs in resp["observations"]:
        mask = obs.get("mask")
        if mask is None:
            continue
        where = f"observation {obs['id']!r} mask {mask['asset']!r}"
        try:
            target = envmanifest.resolve_asset(assets_dir, mask["asset"], where)
        except ManifestError as exc:
            raise BackendError(str(exc)) from exc
        image = cv2.imread(str(target), cv2.IMREAD_UNCHANGED) if target.is_file() else None
        if image is None:
            raise BackendError(f"{where} is missing or not a readable PNG")
        if image.ndim != 2 or image.shape != shape:
            raise BackendError(
                f"{where} is {image.shape[1]}x{image.shape[0]} "
                f"({'single' if image.ndim == 2 else 'multi'}-channel) and does not match the "
                f"{shape[1]}x{shape[0]} source frame; masks must be single-channel PNG image files"
            )
        ys, xs = np.nonzero(image)
        if xs.size == 0:
            raise BackendError(f"{where}: the mask is empty")
        x0, y0, x1, y1 = obs["bbox"]
        if xs.min() < x0 or ys.min() < y0 or xs.max() + 1 > x1 or ys.max() + 1 > y1:
            raise BackendError(f"{where} has pixels outside its bbox {obs['bbox']}")


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


def _geometry_block(
    mode: str, backend: Backend, resp: dict | None, unavailable_reason: str | None = None
) -> dict:
    if mode == "off":
        return {"status": "not_requested", "mode": mode, "reason": "--geometry off"}
    if resp is not None and resp.get("geometry", {}).get("status") in (
        "relative",
        "relative-camera-frame",
        "registered_relative",
        "registered_metric",
    ):
        return {**resp["geometry"], "mode": mode}
    if resp is not None and resp.get("geometry", {}).get("status") == "unavailable":
        return {**resp["geometry"], "mode": mode}
    if resp is None or not backend.supports_geometry():
        reason = unavailable_reason or f"backend {backend.name!r} has no geometry support"
        return {
            "status": "unavailable",
            "mode": mode,
            "reason": reason,
        }
    return {**resp["geometry"], "mode": mode}


def _geometry_filter(
    resp: dict,
    *,
    pose_records: list[dict] | None = None,
    frame_ids: list[int] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Separate static geometry candidates from conservatively excluded tracks."""
    entities = {entity["id"]: entity for entity in resp.get("entities", [])}
    observations_by_entity: dict[str, list[dict]] = {identifier: [] for identifier in entities}
    for observation in resp.get("observations", []):
        observations_by_entity.setdefault(observation["entity"], []).append(observation)

    static_ids = []
    excluded = []
    masks = []
    for identifier, entity in entities.items():
        motion = entity["motion"]
        observations = observations_by_entity.get(identifier, [])
        if motion == "static":
            if observations:
                static_ids.append(identifier)
            continue
        if motion not in ("dynamic", "unknown"):
            continue
        masked_frames = []
        bbox_frames = []
        unmasked_frames = []
        for observation in observations:
            frame_id = observation["frame_id"]
            mask = observation.get("mask")
            if isinstance(mask, dict) and isinstance(mask.get("asset"), str):
                masks.append({"frame_id": frame_id, "asset": mask["asset"]})
                masked_frames.append(frame_id)
            elif isinstance(observation.get("bbox"), list):
                masks.append({"frame_id": frame_id, "bbox": observation["bbox"]})
                bbox_frames.append(frame_id)
            else:
                unmasked_frames.append(frame_id)
        excluded.append(
            {
                "id": identifier,
                "motion": motion,
                "reason": (
                    "dynamic entity excluded from static geometry"
                    if motion == "dynamic"
                    else "unknown motion excluded conservatively from static geometry"
                ),
                "observed_frame_ids": [item["frame_id"] for item in observations],
                "masked_frame_ids": masked_frames,
                "bbox_frame_ids": bbox_frames,
                "unmasked_frame_ids": unmasked_frames,
            }
        )
    skeletons: dict[int, dict] = {}
    if pose_records is not None:
        if width is None or height is None:
            raise ValueError("source dimensions are required when filtering supplied poses")
        wanted_frames = set(frame_ids) if frame_ids is not None else None
        for record in pose_records:
            frame_id = record["frame_id"]
            if wanted_frames is not None and frame_id not in wanted_frames:
                continue
            for detection in record["detections"]:
                tracking_id = detection["tracking_id"]
                item = skeletons.setdefault(
                    tracking_id,
                    {
                        "id": f"nim-skeleton:{tracking_id}",
                        "tracking_id": tracking_id,
                        "reason": "supplied NIM skeleton bbox excluded from static geometry",
                        "observed_frame_ids": [],
                        "bbox_frame_ids": [],
                        "unmasked_frame_ids": [],
                    },
                )
                item["observed_frame_ids"].append(frame_id)
                x, y, box_width, box_height = detection["bbox"]
                bbox = [
                    max(0.0, float(x)),
                    max(0.0, float(y)),
                    min(float(width), float(x + box_width)),
                    min(float(height), float(y + box_height)),
                ]
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    masks.append({"frame_id": frame_id, "bbox": bbox, "source": "nim_skeleton"})
                    item["bbox_frame_ids"].append(frame_id)
                else:
                    item["unmasked_frame_ids"].append(frame_id)
    skeleton_exclusion = (
        {
            "status": "unavailable",
            "reason": "no supplied pose input; person exclusion from geometry is unavailable",
        }
        if pose_records is None
        else {
            "status": "applied",
            "reason": "supplied pose boxes excluded from static geometry at sampled frames",
        }
    )
    return {
        "static_entity_ids": sorted(static_ids),
        "excluded_entities": excluded,
        "excluded_skeletons": [skeletons[key] for key in sorted(skeletons)],
        "skeleton_exclusion": skeleton_exclusion,
        "excluded_pixel_regions": masks,
    }


def run_cli(args) -> int:
    video = args.video
    out = Path(args.out or os.path.splitext(video)[0] + ".environment.json")
    calibration_path = Path(args.calibration).resolve() if args.calibration else None
    overlay_arg = getattr(args, "overlay", None)
    overlay_out = Path(overlay_arg).resolve() if overlay_arg else None
    out_path = out.resolve()
    assets_out = envmanifest.assets_dir_for(out).resolve()
    video_path = Path(video).resolve()
    if out_path == video_path:
        _fail_options("--out would overwrite the source video")
    if video_path.is_relative_to(assets_out):
        _fail_options("--out asset bundle would remove the video input")
    if calibration_path is not None:
        if calibration_path == out_path:
            _fail_options("--out would overwrite the calibration input")
        if calibration_path.is_relative_to(assets_out):
            _fail_options("--out asset bundle would remove the calibration input")
    if args.poses:
        poses_path = Path(args.poses).resolve()
        if poses_path == out_path:
            _fail_options("--out would overwrite the pose input")
        if poses_path.is_relative_to(assets_out):
            _fail_options("--out asset bundle would remove the poses input")
    if args.viewer:
        from .environment_viewer import validate_viewer_paths

        try:
            validate_viewer_paths(out, video, args.viewer, args.poses, calibration_path)
        except ManifestError as exc:
            _fail_options(str(exc))
    viewer_out = Path(args.viewer).resolve() if args.viewer else None
    viewer_bundle = viewer_out.with_name(f"{viewer_out.stem}.viewer.assets") if viewer_out else None
    try:
        validate_artifact_paths(
            inputs=[
                ("source video", video),
                ("pose file", args.poses),
                ("calibration file", getattr(args, "calibration", None)),
            ],
            files=[
                ("environment manifest", out),
                ("environment overlay", overlay_out),
                ("viewer page", viewer_out),
            ],
            directories=[
                ("environment assets", envmanifest.assets_dir_for(out)),
                ("viewer asset bundle", viewer_bundle),
            ],
        )
    except ManifestError as exc:
        _fail_options(str(exc))
    vocabulary, backend, device = preflight_options(args)
    info = _probe(video)
    if args.sample_fps > float(info["rate"]):
        _fail_options(
            f"--sample-fps {args.sample_fps} exceeds the clip's frame rate {float(info['rate']):g}"
        )
    source_sha256 = artifacts.sha256_file(video)
    pose_block, pose_records = (
        _load_pose_join(args.poses, info, source_sha256) if args.poses else (None, None)
    )
    clock = artifacts.FrameClock(info["rate"].numerator, info["rate"].denominator)
    frame_ids = clock.sample(info["frame_count"], args.sample_fps)
    try:
        backend_limit = backend.max_frames()
    except BackendUnavailable as exc:
        _fail_options(str(exc))
    limit = min(MAX_SAMPLED_FRAMES, backend_limit or MAX_SAMPLED_FRAMES)
    if len(frame_ids) > limit:
        _fail_options(
            f"{len(frame_ids)} frames would be scanned; the limit for backend "
            f"{backend.name!r} is {limit}. Lower --sample-fps or use a shorter clip"
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
    try:
        geometry_reference = envgeometry.load_reference(
            args.calibration,
            width=info["width"],
            height=info["height"],
            frame_count=info["frame_count"],
        )
    except envgeometry.GeometryReferenceError as exc:
        _fail_options(str(exc))

    if args.geometry == "required":
        validate_required_reference(geometry_reference, backend)

    geometry_worker = envworkers.discover_geometry() if args.geometry != "off" else None
    geometry_unavailable_reason = None
    geometry_device = None
    geometry_identity = None
    if args.geometry != "off":
        if geometry_worker is None:
            geometry_unavailable_reason = (
                "DA3 geometry worker is not installed; run `just da3-geometry-setup`"
            )
        else:
            try:
                geometry_devices = geometry_worker.available_devices()
                if args.device == "auto":
                    geometry_device = geometry_devices[0] if geometry_devices else None
                elif args.device in geometry_devices:
                    geometry_device = args.device
                else:
                    geometry_unavailable_reason = (
                        f"device {args.device!r} is not available for DA3 geometry "
                        f"(available: {', '.join(geometry_devices) or 'none'})"
                    )
                if geometry_device is None and geometry_unavailable_reason is None:
                    geometry_unavailable_reason = "DA3 geometry worker reports no usable device"
                geometry_identity = geometry_worker.cache_identity()
                geometry_limit = geometry_worker.max_frames()
            except BackendUnavailable as exc:
                geometry_unavailable_reason = str(exc)
                geometry_worker = None
                geometry_identity = {"status": "unavailable", "reason": geometry_unavailable_reason}
            if (
                geometry_worker is not None
                and geometry_device is not None
                and geometry_limit is not None
                and len(frame_ids) > geometry_limit
            ):
                frame_word = "frame" if geometry_limit == 1 else "frames"
                geometry_unavailable_reason = (
                    f"DA3 geometry limit is {geometry_limit} sampled {frame_word}; "
                    f"this request has {len(frame_ids)}"
                )
                geometry_device = None
                if args.geometry == "required":
                    _fail_options(geometry_unavailable_reason)
    if args.geometry == "required" and geometry_worker is None and not backend.supports_geometry():
        _fail_options(
            geometry_unavailable_reason
            or "--geometry required, but no independent geometry worker is available"
        )

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
        "geometry_reference": geometry_reference,
        "geometry_device": geometry_device,
        "requested_labels": [v["label"] for v in vocabulary if v["source"] == "user"],
        "label_vocabulary": vocabulary,
        "source_sha256": source_sha256,
        "poses": pose_block,
        "source": {k: source[k] for k in ("width", "height", "frame_rate", "frame_count")},
        "shots": [{k: sh[k] for k in ("id", "first_frame", "last_frame")} for sh in shot_list],
        "frames": frames,
    }
    print(
        f"environment: backend {backend.name} on {device}, {len(frames)} frames "
        f"(~{args.sample_fps:g} fps, geometry {args.geometry})"
    )

    cache_root = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    perception_identity = None if args.no_cache else backend.cache_identity()
    if args.no_cache or perception_identity is None:
        identity = None
    else:
        identity = {"perception": perception_identity}
        if args.geometry != "off":
            identity["geometry"] = geometry_identity or {
                "status": "unavailable",
                "reason": geometry_unavailable_reason,
            }
    key = cache_key(identity, request) if identity is not None else None

    with artifacts.staging_dir(out) as stage:
        assets_stage = stage / "assets"
        assets_stage.mkdir()
        try:
            resp = _cache_get(cache_root, key, assets_stage) if key is not None else None
            if resp is not None and key is not None:
                print(f"environment: cache hit ({key[:12]})")
                _validate_response(resp, request, args.geometry)
                _check_masks(resp, request, assets_stage)
                assets = _collect_assets(assets_stage)
            else:
                if key is not None:
                    print(f"environment: cache miss ({key[:12]})")
                resp = backend.run(request, assets_stage)
                _validate_response(resp, request, "auto")
                _check_masks(resp, request, assets_stage)
                if args.geometry == "off":
                    resp["geometry"] = {
                        "status": "not_requested",
                        "reason": "--geometry off",
                    }
                elif geometry_worker is not None and geometry_device is not None:
                    geometry_request = {
                        **request,
                        "device": geometry_device,
                        "geometry_filter": _geometry_filter(
                            resp,
                            pose_records=pose_records,
                            frame_ids=frame_ids,
                            width=info["width"],
                            height=info["height"],
                        ),
                    }
                    try:
                        result = geometry_worker.run(geometry_request, assets_stage)
                        if not isinstance(result, dict) or result.get("status") not in (
                            "relative",
                            "relative-camera-frame",
                            "unavailable",
                        ):
                            raise BackendError("DA3 worker returned an invalid geometry status")
                        if geometry_reference is not None:
                            try:
                                result = envregistration.register_geometry(
                                    result, geometry_reference, assets_stage
                                )
                            except envregistration.RegistrationError as exc:
                                result["reason"] = (
                                    f"{result.get('reason', 'DA3 geometry is relative')}; "
                                    f"metric registration abstained: {exc}"
                                )
                                if args.geometry == "required" and not backend.supports_geometry():
                                    raise BackendError(
                                        f"--geometry required, no valid shared registration: {exc}"
                                    ) from exc
                        resp["geometry"] = result
                    except (BackendError, BackendUnavailable) as exc:
                        resp["geometry"] = {
                            "status": "unavailable",
                            "reason": f"DA3 geometry inference failed: {exc}",
                        }
                elif backend.supports_geometry():
                    pass
                else:
                    resp["geometry"] = {
                        "status": "unavailable",
                        "reason": geometry_unavailable_reason
                        or f"backend {backend.name!r} has no geometry support",
                    }
                _validate_response(resp, request, args.geometry)
                _check_masks(resp, request, assets_stage)
                assets = _collect_assets(assets_stage)
                if key is not None and identity is not None and resp["status"] == "complete":
                    _cache_put(cache_root, key, identity, resp, assets_stage, assets)
        except KeyboardInterrupt:
            print("error: interrupted; no environment manifest was written", file=sys.stderr)
            return EXIT_INTERRUPTED
        except BackendError as exc:
            die(f"environment scan failed: {exc}")
        except Exception as exc:
            die(f"environment scan failed: backend {backend.name!r} raised {exc!r}")

        resp.setdefault("perception_status", resp["status"])
        # Apply abstention semantics after both cache hits and fresh inference so cached
        # relative geometry cannot be mistaken for a complete shared-frame scan.
        if args.geometry == "auto" and resp["geometry"]["status"] not in REGISTERED:
            geometry_reason = resp["geometry"].get("reason", "registration unavailable")
            prefix = "geometry registration unavailable: "
            if resp.get("status") == "partial":
                reason = resp.get("reason", "backend returned a partial result")
                if prefix not in reason:
                    reason = f"{reason}; {prefix}{geometry_reason}"
            else:
                reason = f"{prefix}{geometry_reason}"
            resp["status"] = "partial"
            resp["reason"] = reason

        manifest_pose_block = pose_block
        pose_manifest_path = getattr(args, "pose_manifest_path", None)
        if pose_block is not None and pose_manifest_path is not None:
            manifest_pose_block = {**pose_block, "path": str(Path(pose_manifest_path).resolve())}
        manifest_source = source
        source_manifest_path = getattr(args, "source_manifest_path", None)
        if source_manifest_path is not None:
            manifest_source = {**source, "path": str(Path(source_manifest_path).resolve())}
        manifest = {
            "schema": SCHEMA_VERSION,
            "run": {
                "status": resp["status"],
                "perception_status": resp.get("perception_status", resp["status"]),
                **({"reason": resp["reason"]} if resp["status"] != "complete" else {}),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tool": {"name": "skeleton-maker", "version": __version__},
            },
            "source": manifest_source,
            "processed_frames": [
                {"frame_id": i, "time": clock.time_pair(i), "time_s": float(clock.time(i))}
                for i in frame_ids
            ],
            "poses": manifest_pose_block,
            "frame_range": [frame_ids[0], frame_ids[-1]],
            "shots": shot_list,
            "shot_detection": detection,
            "entities": resp["entities"],
            "observations": resp["observations"],
            "relations": labels.containment(resp["entities"], resp["observations"]),
            "geometry": {
                **_geometry_block(args.geometry, backend, resp, geometry_unavailable_reason),
                **(
                    {
                        "calibration": geometry_reference,
                        "calibration_source": os.path.abspath(args.calibration),
                    }
                    if geometry_reference is not None
                    else {}
                ),
            },
            "backend": {**resp["backend"], "device": resp["device"]},
            "config": {
                "requested_labels": request["requested_labels"],
                "label_vocabulary": vocabulary,
                "geometry_mode": args.geometry,
                "sample_fps": args.sample_fps,
                "device": device,
                **({"calibration": geometry_reference} if geometry_reference is not None else {}),
            },
            "assets": assets,
        }
        try:
            envmanifest.validate_manifest(manifest)
            envmanifest.check_assets(manifest, assets_stage)
        except ManifestError as exc:
            die(f"environment scan failed: backend output is invalid: {exc}")
        text = envmanifest.dumps(manifest)
        if args.viewer or overlay_out:
            from . import environment_overlay, environment_viewer

            stage_manifest = stage / out.name
            stage_environment_assets = envmanifest.assets_dir_for(stage_manifest)
            try:
                from contextlib import ExitStack

                with ExitStack() as stack:
                    viewer_stage = (
                        stack.enter_context(artifacts.staging_dir(viewer_out))
                        if viewer_out is not None
                        else None
                    )
                    overlay_stage = (
                        stack.enter_context(artifacts.staging_dir(overlay_out))
                        if overlay_out is not None
                        else None
                    )
                    artifacts.publish(
                        stage_manifest,
                        text,
                        assets_stage if assets else None,
                        stage_environment_assets,
                    )
                    outputs = [
                        (stage_manifest, out),
                        (
                            stage_environment_assets if assets else None,
                            envmanifest.assets_dir_for(out),
                        ),
                    ]
                    if (
                        viewer_out is not None
                        and viewer_bundle is not None
                        and viewer_stage is not None
                    ):
                        staged_viewer_page = viewer_stage / viewer_out.name
                        staged_viewer_bundle = viewer_stage / viewer_bundle.name
                        environment_viewer.prepare_viewer(
                            stage_manifest,
                            video,
                            viewer_out,
                            staged_html=staged_viewer_page,
                            staged_bundle=staged_viewer_bundle,
                            pose_path=args.poses,
                        )
                        outputs.extend(
                            [
                                (staged_viewer_page, viewer_out),
                                (staged_viewer_bundle, viewer_bundle),
                            ]
                        )
                    if overlay_out is not None and overlay_stage is not None:
                        staged_overlay = overlay_stage / overlay_out.name
                        environment_overlay.render_overlay(
                            video,
                            stage_manifest,
                            staged_overlay,
                            pose_path=args.poses,
                        )
                        outputs.append((staged_overlay, overlay_out))
                    cleanup_warnings = artifacts.publish_group(outputs)
                    for warning in cleanup_warnings:
                        print(f"warning: outputs were published, but {warning}", file=sys.stderr)
            except (
                ManifestError,
                OSError,
                environment_overlay.OverlayError,
                poses.PoseFileError,
            ) as exc:
                die(f"environment viewer failed: {exc}", EXIT_FAILED)
        else:
            artifacts.publish(
                out, text, assets_stage if assets else None, envmanifest.assets_dir_for(out)
            )

    _print_summary(manifest, out)
    if args.viewer:
        print(f"  viewer  : {viewer_out} (+ local companion bundle {viewer_bundle})")
    if overlay_out is not None:
        print(f"  overlay : {overlay_out}")
    run = manifest["run"]
    return (
        EXIT_PARTIAL
        if run["status"] == "partial" and run.get("perception_status") != "complete"
        else 0
    )


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
    run = manifest["run"]
    if status == "partial" and run.get("perception_status") == "complete":
        print(
            f"warning: semantic scan completed, but spatial registration is partial: "
            f"{geometry['reason']}. No validated shared metric world is available.",
            file=sys.stderr,
        )
    elif status != "complete":
        print(
            f"warning: this scan is {status}: {manifest['run']['reason']}. "
            f"Exiting {EXIT_PARTIAL} so scripts do not mistake it for a complete run.",
            file=sys.stderr,
        )
