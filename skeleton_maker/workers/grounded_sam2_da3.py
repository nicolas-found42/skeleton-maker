# SPDX-License-Identifier: MIT
"""Environment worker: Grounded DINO detections tracked with SAM 2.1 video propagation.

This file runs under the *worker's* interpreter (see docs/environment-worker.md), never inside
the base package, so it may import torch and friends. Protocol (``skeleton_maker.envworkers``)::

    python grounded_sam2_da3.py preflight
    python grounded_sam2_da3.py run REQUEST.json ASSETS_DIR RESPONSE.json

Grounding DINO detects current-frame entrants. SAM 2 video state propagates prompted masks within
each visual shot and supplies the evidence used to associate detections across sampled frames.
Unconfirmed propagation is not emitted as an observed trajectory. Geometry is not provided.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

WORKER_VERSION = "2"
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

#: Used only when a request carries no label_vocabulary. The command always sends one.
DEFAULT_VOCABULARY = [
    {"label": label, "family": family}
    for family, labels in (
        ("surface", ("wall", "floor", "ceiling")),
        ("object", ("chair", "table", "bottle", "cup")),
        ("vehicle", ("car", "truck", "bus", "bicycle", "motorcycle")),
    )
    for label in labels
]
#: Prompted one at a time: in a group, Grounding DINO tends to lose these.
PROMPTED_ALONE = ("wall", "floor", "ceiling")
ALIASES = {
    "couch": "sofa",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "television": "monitor",
    "tv": "monitor",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "lorry": "truck",
    "automobile": "car",
}
TRACK_ASSOCIATION = "sam2_video_mask_coverage"
TRACK_MIN_OVERLAP = 0.35
TRACK_AMBIGUITY_MARGIN = 0.12
MOTION_METHOD = "median_background_flow_residual"
MOTION_THRESHOLD_DIAGONALS = 0.02
MOTION_FLOW_DISPERSION_THRESHOLD_DIAGONALS = 0.02

SETTINGS = {
    "box_threshold": 0.25,
    "text_threshold": 0.2,
    "nms_iou": 0.7,
    "max_boxes_per_object_label": 6,
    "max_boxes_per_surface_label": 8,
    "max_unknown_per_frame": 4,
    "object_prompt_group_size": 4,
    "preprocessing": "grounding-dino HF processor defaults on RGB frames at source resolution",
    "labels": "request-vocabulary",
    "prompted_alone": list(PROMPTED_ALONE),
    "tracking": {
        "predictor": "SAM2VideoPredictor per shot, one internal video frame per sampled source frame",
        "association": TRACK_ASSOCIATION,
        "minimum_mask_coverage": TRACK_MIN_OVERLAP,
        "ambiguity_margin": TRACK_AMBIGUITY_MARGIN,
        "motion_method": MOTION_METHOD,
        "motion_threshold_frame_diagonals": MOTION_THRESHOLD_DIAGONALS,
        "camera_flow_dispersion_threshold_frame_diagonals": MOTION_FLOW_DISPERSION_THRESHOLD_DIAGONALS,
        "motion_calibrated": False,
    },
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


def _resolve_phrase(phrase: str, prompt_labels: list[str]) -> tuple[str, str, list[str]]:
    """``(status, normalized, candidates)`` for a phrase Grounding DINO returned.

    ``matched`` when the phrase is one prompted label (or an alias of one). A phrase that spans
    several prompted labels is ``ambiguous`` and lists them; anything else is ``unknown``.
    Neither is ever guessed into a class.
    """
    phrase = phrase.strip().lower()
    if phrase in prompt_labels:
        return "matched", phrase, []
    if ALIASES.get(phrase) in prompt_labels:
        return "matched", ALIASES[phrase], []
    words = phrase.split()
    spanned = [label for label in prompt_labels if label in phrase or label in words]
    if len(spanned) >= 2:
        return "ambiguous", "unknown", spanned
    return "unknown", "unknown", []


def _read_frames(video: str, frame_ids: list[int]) -> Iterator[tuple[int, np.ndarray | None]]:
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


def _nms(torchvision, torch, group):
    boxes = torch.tensor([f["box"] for f in group])
    scores = torch.tensor([f["score"] for f in group])
    return torchvision.ops.nms(boxes, scores, SETTINGS["nms_iou"]).tolist()


def _detect(proc, model, image, prompt_labels, device, torch):
    """Run one Grounding DINO prompt; returns one dict per box (status, label, phrase, ...)."""
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
        if not phrase.strip():
            continue
        status, label, candidates = _resolve_phrase(phrase, prompt_labels)
        found.append(
            {
                "status": status,
                "label": label,
                "phrase": phrase,
                "candidates": candidates,
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return found


def _box_for_mask(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _mask_box_coverage(mask, box):
    """Fraction of a detector box covered by the propagated binary mask pixels."""
    import numpy as np

    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(mask.shape[1], int(np.ceil(box[2])))
    y1 = min(mask.shape[0], int(np.ceil(box[3])))
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return float(np.count_nonzero(mask[y0:y1, x0:x1]) / area) if area else 0.0


def _associate_detections(propagated, detections):
    """Match known-label detections against the prior SAM2-propagated track masks.

    The score is the fraction of each detector box covered by the propagated mask pixels. It is a
    deterministic association heuristic, not an identity probability. Close competing tracks
    and low-overlap candidates remain unassigned so they can seed a new identity conservatively.
    """
    predictions = []
    for obj_id, track in propagated.items():
        if track["mask"].any() and track.get("label") not in (None, "unknown"):
            predictions.append((obj_id, track))
    ranked = {}
    for index, det in enumerate(detections):
        candidates = []
        if det.get("status") == "matched" and det.get("label") != "unknown":
            for obj_id, track in predictions:
                if track.get("label") != det["label"]:
                    continue
                candidates.append(
                    (_mask_box_coverage(track["mask"], det["box"]), obj_id, track["entity"])
                )
        ranked[index] = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[2]))

    result = {}
    used = set()
    order = sorted(
        ranked,
        key=lambda index: ranked[index][0][0] if ranked[index] else 0.0,
        reverse=True,
    )
    for index in order:
        candidates = ranked[index]
        if not candidates:
            result[index] = {
                "entity": None,
                "score": 0.0,
                "method": TRACK_ASSOCIATION,
                "uncertain_candidates": [],
            }
            continue
        score, obj_id, entity_id = candidates[0]
        close = [
            candidate[2]
            for candidate in candidates
            if score - candidate[0] < TRACK_AMBIGUITY_MARGIN
        ]
        if score < TRACK_MIN_OVERLAP or len(close) > 1 or obj_id in used:
            result[index] = {
                "entity": None,
                "score": round(float(score), 3),
                "method": TRACK_ASSOCIATION,
                "uncertain_candidates": close or [entity_id],
            }
            continue
        used.add(obj_id)
        result[index] = {
            "entity": entity_id,
            "object_id": obj_id,
            "score": round(float(score), 3),
            "method": TRACK_ASSOCIATION,
            "uncertain_candidates": [],
        }
    return result


def _motion_from_centers(observations, camera_flows, diagonal):
    """Return a camera-translation-compensated motion label from at least two observations."""
    if len(observations) < 2 or len(camera_flows) < len(observations) - 1 or diagonal <= 0:
        return "unknown"
    if any(
        len(flow) > 2 and flow[2] / diagonal > MOTION_FLOW_DISPERSION_THRESHOLD_DIAGONALS
        for flow in camera_flows
    ):
        return "unknown"
    residuals = []
    for (first, second), flow in zip(itertools.pairwise(observations), camera_flows, strict=False):
        displacement = (
            second["center"][0] - first["center"][0] - flow[0],
            second["center"][1] - first["center"][1] - flow[1],
        )
        residuals.append((displacement[0] ** 2 + displacement[1] ** 2) ** 0.5 / diagonal)
    residual = sorted(residuals)[len(residuals) // 2]
    return "dynamic" if residual > MOTION_THRESHOLD_DIAGONALS else "static"


def _mark_track_missing(
    track,
    frame_id,
    reason="no detector box confirmed the propagated mask; occlusion vs missed detection is unknown",
):
    """Record a detector-missed sampled frame without synthesizing an observation."""
    if not track["gaps"] or track["gaps"][-1].get("reacquired_frame") is not None:
        track["gaps"].append(
            {
                "sampled_frame_ids": [frame_id],
                "reacquired_frame": None,
                "status": "lost",
                "reason": reason,
            }
        )
    else:
        track["gaps"][-1]["sampled_frame_ids"].append(frame_id)
    track["status"] = "lost"


def _mark_track_observed(track, frame_id):
    """Record an observed/reacquired sample only after a confirmed SAM2 mask is available."""
    if track["gaps"] and track["gaps"][-1].get("reacquired_frame") is None:
        track["gaps"][-1]["reacquired_frame"] = frame_id
        track["gaps"][-1]["status"] = "reacquired"
    track["last_observed_frame"] = frame_id
    track["status"] = "active"


def run(request_path: str, assets_dir: str, response_path: str) -> int:
    started = time.time()
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    assets = Path(assets_dir)
    frames = request["frames"]
    frame_ids = [f["frame_id"] for f in frames]
    if len(frame_ids) > MAX_FRAMES:
        raise RuntimeError(f"{len(frame_ids)} frames exceeds this worker's limit of {MAX_FRAMES}")
    device = request["device"]
    vocabulary = {
        entry["label"]: entry["family"]
        for entry in request.get("label_vocabulary") or DEFAULT_VOCABULARY
    }
    alone = [label for label in PROMPTED_ALONE if label in vocabulary]
    grouped = [label for label in vocabulary if label not in alone]

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

    # These model packages are installed only in the optional inference-worker environment.
    from sam2.build_sam import build_sam2_video_predictor  # ty: ignore[unresolved-import]
    from transformers import (  # ty: ignore[unresolved-import]
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    _progress(f"loading models on {device}")
    dino_dir = str(_model_path("grounding-dino"))
    processor = AutoProcessor.from_pretrained(dino_dir)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_dir).to(device).eval()
    sam_info = MODELS["sam2"]
    predictor = build_sam2_video_predictor(
        sam_info["config"],
        str(_model_path("sam2") / sam_info["weights"]),
        device=device,
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
    decoded: dict[str, list[tuple[int, int, np.ndarray]]] = {}
    shot_order = []
    height = width = 0
    with tempfile.TemporaryDirectory(prefix="skeleton-maker-sam2-video-") as temp_dir:
        temp_root = Path(temp_dir)
        for frame_id, rgb in _read_frames(request["video"], frame_ids):
            if rgb is None:
                failed.append(frame_id)
                continue
            height, width = rgb.shape[:2]
            shot = shot_of(frame_id)
            if shot not in decoded:
                decoded[shot] = []
                shot_order.append(shot)
            local_idx = len(decoded[shot])
            shot_dir = temp_root / shot
            shot_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(shot_dir / f"{local_idx:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"cannot stage source frame {frame_id} for SAM2 video tracking")
            decoded[shot].append((frame_id, local_idx, rgb))

        next_object_id = 1
        all_motion_points = {}
        for shot in shot_order:
            samples = decoded[shot]
            if not samples:
                continue
            state = predictor.init_state(
                str(temp_root / shot), offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            tracks: dict[int, dict] = {}
            last_masks: Any = {}
            previous_gray: np.ndarray | None = None
            camera_flows = []
            for local_idx, (frame_id, _, rgb) in enumerate(samples):
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                if previous_gray is not None:
                    flow_output = np.empty((*gray.shape[:2], 2), dtype=np.float32)
                    flow = cv2.calcOpticalFlowFarneback(
                        previous_gray, gray, flow_output, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    camera_flows.append(
                        (
                            float(np.median(flow[..., 0])),
                            float(np.median(flow[..., 1])),
                            float(
                                np.median(
                                    np.hypot(
                                        flow[..., 0] - np.median(flow[..., 0]),
                                        flow[..., 1] - np.median(flow[..., 1]),
                                    )
                                )
                            ),
                        )
                    )
                previous_gray = gray

                image = Image.fromarray(rgb)
                found = []
                for label in alone:
                    found += _detect(processor, dino, image, [label], device, torch)
                for group in _groups(grouped, SETTINGS["object_prompt_group_size"]):
                    found += _detect(processor, dino, image, group, device, torch)
                raw[str(frame_id)] = found

                kept = []
                for label in dict.fromkeys(f["label"] for f in found):
                    group = [f for f in found if f["label"] == label]
                    order = _nms(torchvision, torch, group)
                    if label == "unknown":
                        limit = SETTINGS["max_unknown_per_frame"]
                    elif vocabulary[label] == "surface":
                        limit = SETTINGS["max_boxes_per_surface_label"]
                    else:
                        limit = SETTINGS["max_boxes_per_object_label"]
                    kept += [group[i] for i in order[:limit]]

                # A surface remains one semantic region per label and shot. Objects and vehicles
                # retain individual detections so same-class instances can have distinct IDs.
                surface_groups = {}
                detections = []
                for det in kept:
                    if det["status"] == "matched" and vocabulary[det["label"]] == "surface":
                        surface_groups.setdefault(det["label"], []).append(det)
                    else:
                        detections.append(det)
                for _label, group in surface_groups.items():
                    best = max(group, key=lambda item: item["score"])
                    merged_det = dict(best)
                    merged_det["box"] = [
                        min(item["box"][0] for item in group),
                        min(item["box"][1] for item in group),
                        max(item["box"][2] for item in group),
                        max(item["box"][3] for item in group),
                    ]
                    detections.append(merged_det)

                propagated = {}
                if local_idx > 0 and tracks:
                    for propagated_idx, object_ids, masks in predictor.propagate_in_video(
                        state, start_frame_idx=local_idx - 1, max_frame_num_to_track=1
                    ):
                        if propagated_idx != local_idx:
                            continue
                        mask_array = masks.detach().float().cpu().numpy()
                        for pos, object_id in enumerate(object_ids):
                            if int(object_id) in tracks and pos < len(mask_array):
                                propagated[int(object_id)] = {
                                    "entity": tracks[int(object_id)]["entity"],
                                    "label": tracks[int(object_id)]["label"],
                                    "mask": mask_array[pos].squeeze() > 0,
                                }
                associations = _associate_detections(propagated, detections)
                prompted = []
                for index, det in enumerate(detections):
                    assignment = associations[index]
                    if assignment["entity"] is None:
                        object_id = next_object_id
                        next_object_id += 1
                        entity_index = sum(1 for item in entities.values() if item["shot"] == shot)
                        label_id = det["label"] if det["status"] == "matched" else "unknown"
                        entity_id = f"{shot}/{label_id}-{entity_index}"
                        labels = {
                            "requested": det["label"] if det["status"] == "matched" else None,
                            "native": det["phrase"],
                            "normalized": det["label"],
                            "status": det["status"],
                        }
                        if det["candidates"]:
                            labels["candidates"] = det["candidates"]
                        entity = {
                            "id": entity_id,
                            "shot": shot,
                            "family": vocabulary[det["label"]]
                            if det["status"] == "matched"
                            else "object",
                            "motion": "unknown",
                            "labels": labels,
                            "track": {
                                "method": "sam2_video_propagation",
                                "status": "active",
                                "first_observed_frame": frame_id,
                                "last_observed_frame": frame_id,
                                "gaps": [],
                                "association_uncertainty": [],
                            },
                        }
                        if assignment["uncertain_candidates"]:
                            entity["track"]["association_uncertainty"].append(
                                {
                                    "frame_id": frame_id,
                                    "candidate_entities": assignment["uncertain_candidates"],
                                    "method": assignment["method"],
                                    "score": assignment["score"],
                                    "meaning": "uncalibrated propagated-mask overlap heuristic",
                                }
                            )
                        entities[entity_id] = entity
                        tracks[object_id] = {
                            "entity": entity_id,
                            "label": det["label"],
                            "has_observation": False,
                        }
                        assignment["entity"] = entity_id
                        assignment["object_id"] = object_id
                    else:
                        object_id = assignment["object_id"]
                        entity = entities[assignment["entity"]]
                        track = entity["track"]
                    object_id = assignment["object_id"]
                    with torch.inference_mode():
                        last_masks = predictor.add_new_points_or_box(
                            state,
                            frame_idx=local_idx,
                            obj_id=object_id,
                            box=np.asarray(det["box"], dtype=np.float32),
                        )[2]
                    prompted.append((object_id, det, assignment))

                prompted_masks = {}
                if prompted:
                    mask_array = last_masks.detach().float().cpu().numpy()
                    current_ids = list(state["obj_ids"])
                    for pos, object_id in enumerate(current_ids):
                        if pos < len(mask_array):
                            prompted_masks[int(object_id)] = mask_array[pos].squeeze() > 0

                observed_ids = set()
                prompted_ids = {object_id for object_id, _, _ in prompted}
                for object_id, det, assignment in prompted:
                    mask = prompted_masks.get(object_id)
                    box = _box_for_mask(mask) if mask is not None else None
                    if mask is None or box is None:
                        continue
                    entity_id = assignment["entity"]
                    entity = entities[entity_id]
                    track = entity["track"]
                    if not tracks[object_id]["has_observation"]:
                        track["first_observed_frame"] = frame_id
                    _mark_track_observed(track, frame_id)
                    tracks[object_id]["has_observation"] = True
                    observed_ids.add(object_id)
                    rel = f"masks/f{frame_id:06d}-{entity_id.replace('/', '_')}.png"
                    cv2.imwrite(str(assets / rel), mask.astype(np.uint8) * 255)
                    x0, y0, x1, y1 = box
                    observation = {
                        "id": f"obs-{len(observations)}",
                        "entity": entity_id,
                        "frame_id": frame_id,
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "mask": {"asset": rel},
                        "score": min(max(det["score"], 0.0), 1.0),
                        "score_meaning": SCORE_MEANING,
                        "visibility": "visible",
                        "association": {
                            "method": assignment["method"],
                            "score": assignment["score"],
                            "meaning": "uncalibrated propagated-mask overlap heuristic",
                        },
                    }
                    observations.append(observation)
                    center = (
                        (float(det["box"][0]) + float(det["box"][2])) / 2,
                        (float(det["box"][1]) + float(det["box"][3])) / 2,
                    )
                    all_motion_points.setdefault(entity_id, []).append(
                        {"center": center, "local_idx": local_idx}
                    )

                for object_id in set(tracks) - observed_ids:
                    track_data = tracks[object_id]
                    if not track_data["has_observation"]:
                        entities.pop(track_data["entity"], None)
                        del tracks[object_id]
                        continue
                    track = entities[track_data["entity"]]["track"]
                    reason = (
                        "detector prompt yielded no non-empty SAM2 mask; visibility is unconfirmed"
                        if object_id in prompted_ids
                        else "no detector box confirmed the propagated mask; occlusion vs missed detection is unknown"
                    )
                    _mark_track_missing(track, frame_id, reason)
                    track["association_uncertainty"].append(
                        {
                            "frame_id": frame_id,
                            "candidate_entities": [],
                            "method": TRACK_ASSOCIATION,
                            "score": 0.0,
                            "meaning": reason,
                        }
                    )
                done += 1
                _progress(f"{done}/{len(frame_ids)} frames (source frame {frame_id})")

            for _object_id, track_data in tracks.items():
                entity = entities[track_data["entity"]]
                track = entity["track"]
                if track["gaps"] and track["gaps"][-1].get("reacquired_frame") is None:
                    track["status"] = "lost"
                points = all_motion_points.get(entity["id"], [])
                flows = []
                for first, second in itertools.pairwise(points):
                    dx = dy = dispersion = 0.0
                    for flow in camera_flows[first["local_idx"] : second["local_idx"]]:
                        dx += flow[0]
                        dy += flow[1]
                        dispersion = max(dispersion, flow[2])
                    flows.append((dx, dy, dispersion))
                motion = _motion_from_centers(
                    points, flows, (width * width + height * height) ** 0.5
                )
                entity["motion"] = motion
                entity["motion_evidence"] = {
                    "method": MOTION_METHOD,
                    "observations": len(points),
                    "threshold_frame_diagonals": MOTION_THRESHOLD_DIAGONALS,
                    "camera_flow_dispersion_threshold_frame_diagonals": (
                        MOTION_FLOW_DISPERSION_THRESHOLD_DIAGONALS
                    ),
                    "calibrated": False,
                }

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
