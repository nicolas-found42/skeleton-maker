# SPDX-License-Identifier: MIT
"""Environment worker: Grounded SAM 2 masks (Grounding DINO boxes prompted per label, SAM 2.1 masks).

This file runs under the *worker's* interpreter (see docs/environment-worker.md), never inside
the base package, so it may import torch and friends. Protocol (``skeleton_maker.envworkers``)::

    python grounded_sam2_da3.py preflight
    python grounded_sam2_da3.py run REQUEST.json ASSETS_DIR RESPONSE.json

Scope of this worker version: perception only. Structural surfaces become one semantic entity
per class per shot; objects and vehicles become one entity per detection per scanned frame,
because tracking identities across frames is a later stage. Geometry is not provided.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import time
from pathlib import Path

WORKER_VERSION = "1"
PROTOCOL = 1
BACKEND_CONTRACT = "skeleton-maker.environment-backend/1"

#: Exact versions the outputs were produced and evaluated with. A mismatch is reported by
#: preflight instead of silently producing differently-behaving results.
PINNED_LIBRARIES = {
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "transformers": "5.18.0",
    "sam2": "1.1.0",
}
IMPORT_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "sam2": "sam2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "numpy": "numpy",
}

MODELS = {
    "grounding-dino": {
        "directory": "grounding-dino-tiny",
        "source": "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
        "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "weights": "model.safetensors",
        "sha256": "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3",
        "license": "Apache-2.0",
    },
    "sam2": {
        "directory": "sam2.1",
        "source": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "weights": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        "license": "Apache-2.0",
    },
}

SURFACE_LABELS = ("wall", "floor", "ceiling")
OBJECT_LABELS = ("chair", "table", "bottle", "cup")
VEHICLE_LABELS = ("car", "truck", "bus", "bicycle", "motorcycle")

SETTINGS = {
    "box_threshold": 0.25,
    "text_threshold": 0.2,
    "nms_iou": 0.7,
    "max_boxes_per_object_label": 6,
    "max_boxes_per_surface_label": 8,
    "object_prompt_group_size": 4,
    "preprocessing": "grounding-dino HF processor defaults on RGB frames at source resolution",
    "labels": "minimal-v1",
}
MAX_FRAMES = 600
SCORE_MEANING = "Grounding DINO box confidence (uncalibrated); the SAM 2.1 mask is not scored"


def models_home() -> Path:
    return Path(
        os.environ.get("SKELETON_MAKER_MODELS")
        or Path(os.path.expanduser("~")) / ".cache" / "skeleton-maker" / "models"
    )


def _model_path(key: str) -> Path:
    return models_home() / MODELS[key]["directory"]


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _devices() -> list[str]:
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def identity() -> dict:
    return {
        "worker_version": WORKER_VERSION,
        "libraries": {name: _version(name) for name in PINNED_LIBRARIES},
        "models": {
            key: {k: v for k, v in info.items() if k in ("source", "revision", "sha256", "license")}
            for key, info in MODELS.items()
        },
        "settings": SETTINGS,
    }


def preflight() -> dict:
    missing = []
    for dist, module in IMPORT_NAMES.items():
        if importlib.util.find_spec(module) is None:
            missing.append(f"python package {dist}")
    for dist, pinned in PINNED_LIBRARIES.items():
        have = _version(dist)
        if have is not None and have.split("+")[0] != pinned:
            missing.append(f"{dist}=={pinned} (installed {have})")
    for key, info in MODELS.items():
        weights = _model_path(key) / info["weights"]
        if not weights.is_file():
            missing.append(f"model file {weights}")
    if missing:
        return {
            "protocol": PROTOCOL,
            "ok": False,
            "missing": missing,
            "message": "the Grounded SAM 2 worker environment is incomplete "
            "(see docs/environment-worker.md)",
        }
    return {
        "protocol": PROTOCOL,
        "ok": True,
        "devices": _devices(),
        "supports_geometry": False,
        "max_frames": MAX_FRAMES,
        "identity": identity(),
    }


# --- run -------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(message: str) -> None:
    print(f"progress: {message}", file=sys.stderr, flush=True)


def _family(label: str) -> str:
    if label in SURFACE_LABELS:
        return "surface"
    if label in VEHICLE_LABELS:
        return "vehicle"
    return "object"


def _read_frames(video: str, frame_ids: list[int]):
    """Yield ``(frame_id, RGB array or None)`` for the wanted frames, decoding sequentially."""
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    wanted = set(frame_ids)
    try:
        for index in range(max(frame_ids) + 1):
            if not cap.grab():
                break
            if index in wanted:
                ok, bgr = cap.retrieve()
                wanted.discard(index)
                yield index, (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
    finally:
        cap.release()
    for index in sorted(wanted):
        yield index, None


def _groups(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def _detect(proc, model, image, prompt_labels, device, torch):
    """Run one Grounding DINO prompt; returns ``[(label, phrase, score, box)]``."""
    text = " ".join(f"{label}." for label in prompt_labels)
    inputs = proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    result = proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=SETTINGS["box_threshold"],
        text_threshold=SETTINGS["text_threshold"],
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = result["boxes"].tolist()
    if not boxes:  # transformers reports text_labels == [""] when nothing passes the threshold
        return []
    phrases, scores = result["text_labels"], result["scores"].tolist()
    if not len(boxes) == len(phrases) == len(scores):
        raise RuntimeError(
            f"Grounding DINO returned {len(boxes)} boxes but {len(phrases)} labels and "
            f"{len(scores)} scores"
        )
    found = []
    for box, score, phrase in zip(boxes, scores, phrases, strict=True):
        label = next((lab for lab in prompt_labels if lab == phrase.strip()), None)
        if label is None:  # a phrase spanning several labels is ambiguous: keep it out
            continue
        found.append((label, phrase, float(score), [float(v) for v in box]))
    return found


def run(request_path: str, assets_dir: str, response_path: str) -> int:
    started = time.time()
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    assets = Path(assets_dir)
    frames = request["frames"]
    frame_ids = [f["frame_id"] for f in frames]
    if len(frame_ids) > MAX_FRAMES:
        raise RuntimeError(f"{len(frame_ids)} frames exceeds this worker's limit of {MAX_FRAMES}")
    device = request["device"]
    extra = [lab.strip().lower() for lab in request.get("requested_labels", []) if lab.strip()]
    surfaces = list(SURFACE_LABELS)
    countables = [
        *OBJECT_LABELS,
        *VEHICLE_LABELS,
        *[
            lab
            for lab in dict.fromkeys(extra)
            if lab not in (*SURFACE_LABELS, *OBJECT_LABELS, *VEHICLE_LABELS)
        ],
    ]

    _progress("verifying model files")
    checkpoints = []
    for key, info in MODELS.items():
        weights = _model_path(key) / info["weights"]
        digest = _sha256(weights)
        if digest != info["sha256"]:
            raise RuntimeError(f"{weights} does not match the pinned sha256 {info['sha256']}")
        checkpoints.append(
            {
                "name": f"{key}:{info['weights']}",
                "sha256": digest,
                "license": info["license"],
                "source": info["source"],
            }
        )

    import cv2
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    _progress(f"loading models on {device}")
    dino_dir = str(_model_path("grounding-dino"))
    processor = AutoProcessor.from_pretrained(dino_dir)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_dir).to(device).eval()
    sam_info = MODELS["sam2"]
    sam = SAM2ImagePredictor(
        build_sam2(
            sam_info["config"], str(_model_path("sam2") / sam_info["weights"]), device=device
        )
    )

    def shot_of(frame_id):
        return next(
            s["id"] for s in request["shots"] if s["first_frame"] <= frame_id <= s["last_frame"]
        )

    (assets / "masks").mkdir(parents=True, exist_ok=True)
    (assets / "raw").mkdir(parents=True, exist_ok=True)
    entities: dict[str, dict] = {}
    observations: list[dict] = []
    raw: dict[str, list] = {}
    failed: list[int] = []
    done = 0

    for frame_id, rgb in _read_frames(request["video"], frame_ids):
        if rgb is None:
            failed.append(frame_id)
            continue
        image = Image.fromarray(rgb)
        height, width = rgb.shape[:2]
        found = []
        for label in surfaces:
            found += _detect(processor, dino, image, [label], device, torch)
        for group in _groups(countables, SETTINGS["object_prompt_group_size"]):
            found += _detect(processor, dino, image, group, device, torch)
        raw[str(frame_id)] = [
            {"label": lab, "phrase": phrase, "score": score, "box": box}
            for lab, phrase, score, box in found
        ]

        kept = []
        for label in dict.fromkeys(f[0] for f in found):
            group = [f for f in found if f[0] == label]
            boxes = torch.tensor([f[3] for f in group])
            scores = torch.tensor([f[2] for f in group])
            order = torchvision.ops.nms(boxes, scores, SETTINGS["nms_iou"]).tolist()
            limit = (
                SETTINGS["max_boxes_per_surface_label"]
                if label in SURFACE_LABELS
                else SETTINGS["max_boxes_per_object_label"]
            )
            kept += [group[i] for i in order[:limit]]
        if kept:
            with torch.inference_mode():
                sam.set_image(np.array(image))
                masks, _, _ = sam.predict(
                    box=np.array([k[3] for k in kept], dtype=np.float32), multimask_output=False
                )
            masks = np.asarray(masks).reshape(len(kept), height, width) > 0
        shot = shot_of(frame_id)
        merged: dict[str, tuple] = {}
        for n, (label, phrase, score, _box) in enumerate(kept):
            mask = masks[n]
            if label in SURFACE_LABELS:
                prev = merged.get(label)
                merged[label] = (
                    (prev[0] | mask) if prev else mask,
                    max(score, prev[1]) if prev else score,
                    phrase,
                )
                continue
            merged[f"{label}#{n}"] = (mask, score, phrase)
        for key, (mask, score, phrase) in merged.items():
            label = key.split("#")[0]
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            if label in SURFACE_LABELS:
                entity_id = f"{shot}/surface-{label}"
            else:
                entity_id = f"{shot}/{label}-f{frame_id}-{key.split('#')[1]}"
            entities.setdefault(
                entity_id,
                {
                    "id": entity_id,
                    "shot": shot,
                    "family": _family(label),
                    "motion": "unknown",
                    "labels": {"native": phrase, "normalized": label},
                },
            )
            rel = f"masks/f{frame_id:06d}-{entity_id.replace('/', '_')}.png"
            cv2.imwrite(str(assets / rel), mask.astype(np.uint8) * 255)
            observations.append(
                {
                    "id": f"obs-{len(observations)}",
                    "entity": entity_id,
                    "frame_id": frame_id,
                    "bbox": [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ],
                    "mask": {"asset": rel},
                    "score": min(max(score, 0.0), 1.0),
                    "score_meaning": SCORE_MEANING,
                    "visibility": "visible",
                }
            )
        done += 1
        _progress(f"{done}/{len(frame_ids)} frames (source frame {frame_id})")

    if not done:
        raise RuntimeError("no frame could be decoded")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1024  # bytes vs KiB
    stats = {"wall_seconds": round(time.time() - started, 1), "peak_rss_mb": round(peak_rss_mb)}
    if device == "mps":
        stats["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / 1e6)
    (assets / "raw" / "detections.json").write_text(
        json.dumps({"worker_version": WORKER_VERSION, "settings": SETTINGS, "frames": raw}),
        encoding="utf-8",
    )
    response = {
        "contract": BACKEND_CONTRACT,
        "status": "partial" if failed else "complete",
        "backend": {
            "name": "grounded-sam2-da3",
            "version": WORKER_VERSION,
            "checkpoints": checkpoints,
            "libraries": identity()["libraries"],
            "settings": SETTINGS,
            "run_stats": stats,
        },
        "device": device,
        "entities": list(entities.values()),
        "observations": observations,
        "geometry": {"status": "unavailable", "reason": "this worker version provides no geometry"},
    }
    if failed:
        response["reason"] = f"{len(failed)} sampled frame(s) could not be decoded: {failed}"
    Path(response_path).write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "preflight":
        print(json.dumps(preflight()))
        sys.exit(0)
    if command == "run" and len(sys.argv) == 5:
        sys.exit(run(*sys.argv[2:5]))
    print(
        "usage: grounded_sam2_da3.py preflight | run REQUEST ASSETS_DIR RESPONSE", file=sys.stderr
    )
    sys.exit(2)
