# SPDX-License-Identifier: MIT
"""Optional Depth Anything 3 Small worker for relative depth and camera geometry.

The base package does not import Torch. Install this worker with
``just da3-geometry-setup``; it speaks the subprocess protocol in :mod:`envworkers` and only
loads the model when ``--geometry auto`` actually invokes it.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

WORKER_VERSION = "2"
PROTOCOL = 1
CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_REPOSITORY = "depth-anything/DA3-SMALL"
MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
MODEL_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
MODEL_LICENSE = "Apache-2.0"
PINNED_LIBRARIES = {
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "numpy": "1.26.4",
    "huggingface_hub": "1.33.0",
    "opencv-python-headless": "4.11.0.86",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "einops": "0.8.1",
    "omegaconf": "2.3.1",
    "imageio": "2.37.0",
    "trimesh": "4.6.4",
    "moviepy": "1.0.3",
    "addict": "2.4.0",
}
# Single-shot batch size measured on the 16 GiB M5/MPS reference workstation.
MAX_FRAMES = 16
PROCESS_RES = 504
PROCESS_RES_METHOD = "upper_bound_resize"
SETTINGS = {
    "model": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
    "preprocessing": {
        "input_color": "RGB",
        "process_res": PROCESS_RES,
        "process_res_method": PROCESS_RES_METHOD,
        "crop": None,
        "lens_transform": "identity or OpenCV undistort from supplied calibration",
    },
    "depth_units": "relative_depth",
    "extrinsics": "world-to-camera, DA3/OpenCV convention, per-shot reference frame",
    "input_extrinsics_alignment": "disabled; this worker supplies no input extrinsics",
    "export": "disabled; this worker consumes prediction arrays directly",
}


def worker_home() -> Path:
    return Path(
        os.environ.get("SKELETON_MAKER_WORKER_HOME")
        or Path.home() / ".cache" / "skeleton-maker" / "workers"
    )


def models_home() -> Path:
    return Path(
        os.environ.get("SKELETON_MAKER_MODELS")
        or Path.home() / ".cache" / "skeleton-maker" / "models"
    )


def model_dir() -> Path:
    return models_home() / "da3-small"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def identity() -> dict:
    checkpoint = model_dir() / "model.safetensors"
    return {
        "worker_version": WORKER_VERSION,
        "code": {
            "repository": "https://github.com/ByteDance-Seed/Depth-Anything-3",
            "revision": CODE_REVISION,
            "license": MODEL_LICENSE,
        },
        "checkpoint": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "file": "model.safetensors",
            "sha256": MODEL_SHA256,
            "license": MODEL_LICENSE,
        },
        "libraries": {name: _version(name) for name in (*PINNED_LIBRARIES, "depth-anything-3")},
        "settings": SETTINGS,
        "checkpoint_file_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
    }


def _devices() -> list[str]:
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _peak_rss_bytes() -> int:
    """Return the worker's process high-water resident set in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _mps_memory_snapshot(device: str) -> dict | None:
    """Capture synchronized MPS allocator snapshots; these are not memory high-water marks."""
    if device != "mps":
        return None
    import torch

    torch.mps.synchronize()
    return {
        "tensor_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _unsupported_upstream_feature(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "this DA3 worker only supports prediction arrays; input-pose alignment and exports "
        "are not available"
    )


def _install_inference_only_adapters() -> None:
    """Skip unused upstream exporter/alignment imports with heavy optional dependencies."""
    export_name = "depth_anything_3.utils.export"
    export_spec = importlib.util.find_spec(export_name)
    if export_spec is None:
        raise RuntimeError("pinned DA3 source has no utils.export package")
    export_module = types.ModuleType(export_name)
    export_module.__path__ = list(export_spec.submodule_search_locations or [])
    export_module.__dict__["export"] = _unsupported_upstream_feature
    sys.modules[export_name] = export_module

    alignment_name = "depth_anything_3.utils.pose_align"
    alignment_module = types.ModuleType(alignment_name)
    alignment_module.__dict__["align_poses_umeyama"] = _unsupported_upstream_feature
    sys.modules[alignment_name] = alignment_module


def preflight() -> dict:
    # Force DA3's dictionary-style configuration dependency to exist before inference.
    missing = []
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "depth_anything_3",
        "huggingface_hub",
        "cv2",
        "PIL",
        "addict",
    ):
        if importlib.util.find_spec(package) is None:
            missing.append(f"python package {package}")
    for distribution, pinned in PINNED_LIBRARIES.items():
        installed = _version(distribution)
        if installed != pinned:
            missing.append(f"{distribution}=={pinned} (installed {installed or 'missing'})")
    revision_file = models_home() / "da3-code-commit.txt"
    if (
        not revision_file.is_file()
        or revision_file.read_text(encoding="utf-8").strip() != CODE_REVISION
    ):
        missing.append(f"Depth Anything 3 source marker at {CODE_REVISION}")
    checkpoint = model_dir() / "model.safetensors"
    if not checkpoint.is_file():
        missing.append(f"model file {checkpoint}")
    elif _sha256(checkpoint) != MODEL_SHA256:
        missing.append(f"model file {checkpoint} does not match the pinned SHA-256")
    if not (model_dir() / "config.json").is_file():
        missing.append(f"model config {model_dir() / 'config.json'}")
    if missing:
        return {
            "protocol": PROTOCOL,
            "ok": False,
            "missing": missing,
            "message": "the DA3 geometry worker is incomplete (see docs/environment-worker.md)",
        }
    try:
        import torch

        _install_inference_only_adapters()
        importlib.import_module("depth_anything_3.api")
    except Exception as exc:
        return {
            "protocol": PROTOCOL,
            "ok": False,
            "missing": [f"DA3 inference import: {type(exc).__name__}: {exc}"],
            "message": "the DA3 inference environment cannot import its pinned model code",
        }
    return {
        "protocol": PROTOCOL,
        "ok": True,
        "devices": _devices(),
        "supports_geometry": True,
        "max_frames": MAX_FRAMES,
        "identity": identity(),
        "torch": torch.__version__,
    }


def _read_frames(video: str, frame_ids: list[int]) -> Iterator[tuple[int, np.ndarray | None]]:
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video {video}")
    wanted = set(frame_ids)
    try:
        for frame_id in range(max(frame_ids) + 1):
            if not cap.grab():
                break
            if frame_id in wanted:
                ok, bgr = cap.retrieve()
                wanted.remove(frame_id)
                yield frame_id, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()
    for frame_id in sorted(wanted):
        yield frame_id, None


def _finite_array(array, shape_tail: tuple[int, ...], name: str):
    import numpy as np

    result = np.asarray(array, dtype=np.float32)
    if result.shape[-len(shape_tail) :] != shape_tail or not np.isfinite(result).all():
        raise RuntimeError(f"DA3 returned invalid {name} with shape {result.shape}")
    return result


def _depth_batch(array, *, expected_frames: int):
    """Validate DA3's native ``(N, H, W)`` depth grid without assuming source size."""
    import numpy as np

    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 3:
        raise RuntimeError(f"DA3 returned depth with shape {result.shape}; expected (N, H, W)")
    if result.shape[0] != expected_frames:
        raise RuntimeError(f"DA3 returned {result.shape[0]} depths for {expected_frames} images")
    if result.shape[1] <= 0 or result.shape[2] <= 0:
        raise RuntimeError(f"DA3 returned depth with empty spatial dimensions: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("DA3 returned depth with non-finite values")
    return result


def _intrinsics_in_source_space(intrinsics, *, scale_x: float, scale_y: float):
    """Convert processed-grid intrinsics to undistorted source-frame pixels."""
    result = intrinsics.copy()
    result[0, :] *= 1.0 / scale_x
    result[1, :] *= 1.0 / scale_y
    return result


def _resize_pixel_maps(*, scale_x: float, scale_y: float):
    """Return both pixel maps for DA3's possibly nonuniform image resize."""
    source_to_processed = [
        [scale_x, 0.0, 0.0],
        [0.0, scale_y, 0.0],
        [0.0, 0.0, 1.0],
    ]
    processed_to_source = [
        [1.0 / scale_x, 0.0, 0.0],
        [0.0, 1.0 / scale_y, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return source_to_processed, processed_to_source


def _full_k_undistort_maps(
    source_intrinsics: np.ndarray,
    coefficients: np.ndarray,
    output_intrinsics: np.ndarray,
    size: tuple[int, int],
):
    """Map ideal output pixels through Brown-Conrady distortion into source pixels."""
    import cv2
    import numpy as np

    width, height = size
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    columns = np.arange(width)
    for first_row in range(0, height, 128):
        last_row = min(first_row + 128, height)
        grid_x, grid_y = np.meshgrid(columns, np.arange(first_row, last_row))
        output_pixels = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), np.ones(grid_x.size)], axis=0
        )
        output_rays = np.linalg.solve(output_intrinsics, output_pixels)
        output_xy = output_rays[:2] / output_rays[2]
        normalized_points = np.column_stack([output_xy.T, np.ones(grid_x.size)])
        distorted_xy, _ = cv2.projectPoints(
            normalized_points,
            np.zeros(3),
            np.zeros(3),
            np.eye(3, dtype=np.float64),
            coefficients,
        )
        source_pixels = source_intrinsics @ np.vstack(
            [distorted_xy.reshape(-1, 2).T, np.ones(grid_x.size)]
        )
        source_pixels /= source_pixels[2]
        map_x[first_row:last_row] = source_pixels[0].reshape(last_row - first_row, width)
        map_y[first_row:last_row] = source_pixels[1].reshape(last_row - first_row, width)
    return map_x, map_y


def _mask_to_processed(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
    lens_transform: dict[str, Any],
):
    """Map a source-frame semantic mask onto DA3's native depth grid."""
    import cv2
    import numpy as np

    processed_size = (round(width * scale_x), round(height * scale_y))
    if lens_transform.get("applied"):
        processed_intrinsics = np.asarray(lens_transform["output_intrinsics"], dtype=np.float64)
        processed_intrinsics[0, :] *= scale_x
        processed_intrinsics[1, :] *= scale_y
        map_x, map_y = _full_k_undistort_maps(
            np.asarray(lens_transform["source_intrinsics"], dtype=np.float64),
            np.asarray(lens_transform["source_coefficients"], dtype=np.float64),
            processed_intrinsics,
            processed_size,
        )
        processed = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
    else:
        processed = cv2.resize(mask, processed_size, interpolation=cv2.INTER_NEAREST)
    return np.asarray(processed) > 0


def _exclude_masked_depth(depth, exclusion_mask):
    """Mark excluded moving/unknown pixels as invalid in the native depth grid."""
    import numpy as np

    if exclusion_mask.shape != depth.shape:
        raise RuntimeError(
            f"processed dynamic mask has shape {exclusion_mask.shape}; expected {depth.shape}"
        )
    filtered = depth.copy()
    filtered[exclusion_mask] = np.nan
    return filtered


def _load_exclusion_masks(assets: Path, geometry_filter: dict) -> dict[int, list]:
    import cv2

    by_frame: dict[int, list] = {}
    for item in geometry_filter.get("excluded_pixel_regions", []):
        frame_id = item["frame_id"]
        if isinstance(item.get("asset"), str):
            asset = item["asset"]
            path = (assets / asset).resolve()
            try:
                path.relative_to(assets.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"dynamic mask asset escapes geometry assets: {asset!r}"
                ) from exc
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot decode dynamic exclusion mask {asset!r}")
            by_frame.setdefault(frame_id, []).append({"mask": mask})
        elif isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4:
            by_frame.setdefault(frame_id, []).append({"bbox": item["bbox"]})
        else:
            raise RuntimeError(f"dynamic exclusion for frame {frame_id} has no mask or bbox")
    return by_frame


def _source_exclusion_mask(regions: list[dict], *, width: int, height: int):
    """Combine full-source masks and boxes into the image mask before DA3 resize mapping."""
    import numpy as np

    exclusion = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        if "mask" in region:
            source_mask = region["mask"]
            if source_mask.shape != (height, width):
                raise RuntimeError(
                    f"dynamic mask has shape {source_mask.shape}; expected {(height, width)}"
                )
            exclusion |= (source_mask > 0).astype(np.uint8)
        else:
            x0, y0, x1, y1 = region["bbox"]
            exclusion[
                max(0, int(np.floor(y0))) : min(height, int(np.ceil(y1))),
                max(0, int(np.floor(x0))) : min(width, int(np.ceil(x1))),
            ] = 1
    return exclusion


def _undistort(image, camera: dict):
    import cv2
    import numpy as np

    matrix = np.asarray(camera["intrinsics"], dtype=np.float64)
    distortion = camera["distortion"]
    coefficients = np.asarray(distortion["coefficients"], dtype=np.float64)
    if distortion["model"] == "none":
        return image, matrix, {"model": "none", "applied": False}
    height, width = image.shape[:2]
    corrected, _ = cv2.getOptimalNewCameraMatrix(
        matrix, coefficients, (width, height), 0.0, (width, height)
    )
    map_x, map_y = _full_k_undistort_maps(matrix, coefficients, corrected, (width, height))
    valid = (map_x >= 0.0) & (map_x <= width - 1.0) & (map_y >= 0.0) & (map_y <= height - 1.0)
    if valid.any():
        rows, columns = np.where(valid)
        roi = [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        ]
    else:
        roi = [0, 0, 0, 0]
    return (
        cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0),
        ),
        corrected,
        {
            "model": "opencv-brown-conrady",
            "applied": True,
            "source_intrinsics": matrix.tolist(),
            "source_coefficients": coefficients.tolist(),
            "output_intrinsics": corrected.tolist(),
            "valid_roi": [int(value) for value in roi],
        },
    )


def _as_list(values, shape_tail: tuple[int, ...], name: str) -> list:
    return _finite_array(values, shape_tail, name).tolist()


def run(request_path: str, assets_dir: str, response_path: str) -> int:
    import numpy as np
    from PIL import Image

    _install_inference_only_adapters()
    # Installed only in the optional worker environment; inference imports stay lazy in the base.
    from depth_anything_3.api import DepthAnything3  # ty: ignore[unresolved-import]

    started = time.time()
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    frame_ids = [frame["frame_id"] for frame in request["frames"]]
    if not frame_ids:
        raise RuntimeError("DA3 geometry request has no frames")
    if len(frame_ids) > MAX_FRAMES:
        raise RuntimeError(f"{len(frame_ids)} frames exceeds this worker's limit of {MAX_FRAMES}")
    assets = Path(assets_dir)
    device = request["device"]
    calibration = request.get("geometry_reference") or {}
    geometry_filter = request.get("geometry_filter") or {}
    static_entity_ids = geometry_filter.get("static_entity_ids", [])
    excluded_entities = geometry_filter.get("excluded_entities", [])
    exclusion_masks = _load_exclusion_masks(assets, geometry_filter)
    camera = calibration.get("camera")
    decoded: dict[int, np.ndarray] = {
        frame_id: image
        for frame_id, image in _read_frames(request["video"], frame_ids)
        if image is not None
    }
    missed = [frame_id for frame_id in frame_ids if frame_id not in decoded]
    if missed:
        raise RuntimeError(f"could not decode source frames: {missed[:12]}")

    checkpoint = model_dir()
    print(f"progress: loading DA3-Small on {device}", file=sys.stderr, flush=True)
    model = DepthAnything3.from_pretrained(str(checkpoint)).to(device=device)
    geometry_frames = []
    mps_memory_snapshots = []
    shots = request["shots"]
    for shot in shots:
        shot_frames = [
            frame_id
            for frame_id in frame_ids
            if shot["first_frame"] <= frame_id <= shot["last_frame"]
        ]
        if not shot_frames:
            continue
        rgb = [decoded[frame_id] for frame_id in shot_frames]
        source_height, source_width = rgb[0].shape[:2]
        lens_transform = {"model": "none", "applied": False}
        inference_intrinsics = None
        if camera is not None:
            transformed = [_undistort(image, camera) for image in rgb]
            rgb = [entry[0] for entry in transformed]
            inference_intrinsics = np.stack([entry[1] for entry in transformed]).astype(np.float32)
            lens_transform = transformed[0][2]
        images = [Image.fromarray(image, mode="RGB") for image in rgb]
        prediction = model.inference(
            images,
            intrinsics=inference_intrinsics,
            process_res=PROCESS_RES,
            process_res_method=PROCESS_RES_METHOD,
        )
        mps_snapshot = _mps_memory_snapshot(device)
        if mps_snapshot is not None:
            mps_memory_snapshots.append(mps_snapshot)
        depths = _depth_batch(prediction.depth, expected_frames=len(shot_frames))
        intrinsics = _finite_array(prediction.intrinsics, (3, 3), "intrinsics")
        extrinsics = _finite_array(prediction.extrinsics, (3, 4), "extrinsics")
        processed_height, processed_width = depths.shape[1:]
        scale_x = processed_width / source_width
        scale_y = processed_height / source_height
        undistorted_source_to_processed, processed_to_undistorted_source = _resize_pixel_maps(
            scale_x=scale_x, scale_y=scale_y
        )
        for offset, frame_id in enumerate(shot_frames):
            relative = Path("geometry") / "depth" / f"depth-{frame_id:08d}.npy"
            destination = assets / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as stream:
                depth = depths[offset].copy()
                source_regions = exclusion_masks.get(frame_id, [])
                if source_regions:
                    exclusion = _source_exclusion_mask(
                        source_regions, width=source_width, height=source_height
                    )
                    processed_exclusion = _mask_to_processed(
                        exclusion,
                        width=source_width,
                        height=source_height,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        lens_transform=lens_transform,
                    )
                    depth = _exclude_masked_depth(depth, processed_exclusion)
                np.save(stream, depth, allow_pickle=False)
            source_intrinsics = _intrinsics_in_source_space(
                intrinsics[offset], scale_x=scale_x, scale_y=scale_y
            )
            geometry_frames.append(
                {
                    "frame_id": frame_id,
                    "shot": shot["id"],
                    "depth_asset": relative.as_posix(),
                    "intrinsics": source_intrinsics.tolist(),
                    "extrinsics_w2c": extrinsics[offset].tolist(),
                    "depth_pixel_space": "processed_frame_pixels",
                    "preprocessing": {
                        "process_res": PROCESS_RES,
                        "process_res_method": PROCESS_RES_METHOD,
                        "source_size": [source_width, source_height],
                        "processed_size": [processed_width, processed_height],
                        "undistorted_source_to_processed": undistorted_source_to_processed,
                        "processed_to_undistorted_source": processed_to_undistorted_source,
                        "depth_resampling": "none; native DA3 processed depth grid",
                        "intrinsics_pixel_space": "undistorted_source_frame_pixels",
                        "crop": None,
                        "lens_transform": lens_transform,
                    },
                }
            )
            print(f"progress: DA3 geometry frame {frame_id}", file=sys.stderr, flush=True)

    geometry = {
        "status": "relative-camera-frame",
        "reason": "DA3 depth and poses are camera-frame estimates without a solved metric scale",
        "units": "relative_depth",
        "coordinate_convention": (
            "DA3 camera frame: +x right, +y down, +z forward; extrinsics are world-to-camera; "
            "each shot has an independent DA3 reference frame"
        ),
        "scale_provenance": {
            "kind": "relative_model_prediction",
            "metric": False,
            "anchors_recorded_only": bool(calibration.get("anchors")),
        },
        "static_fusion_entities": sorted(static_entity_ids),
        "excluded_dynamic_entities": excluded_entities,
        "excluded_skeletons": geometry_filter.get("excluded_skeletons", []),
        "skeleton_exclusion": geometry_filter.get(
            "skeleton_exclusion",
            {
                "status": "unavailable",
                "reason": "geometry worker request omitted skeleton exclusion metadata",
            },
        ),
        "frames": geometry_frames,
        "backend": identity(),
        "device": device,
        "run_stats": {
            "wall_seconds": round(time.time() - started, 3),
            "peak_rss_bytes": _peak_rss_bytes(),
            **(
                {"mps_memory_after_inference_by_shot": mps_memory_snapshots}
                if mps_memory_snapshots
                else {}
            ),
        },
    }
    Path(response_path).write_text(json.dumps(geometry, allow_nan=False), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["preflight"]:
        print(json.dumps(preflight(), allow_nan=False))
        return 0
    if len(args) == 4 and args[0] == "run":
        return run(*args[1:])
    print(
        "usage: da3_geometry.py preflight | run REQUEST.json ASSETS_DIR RESPONSE.json",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
