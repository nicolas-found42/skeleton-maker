# SPDX-License-Identifier: MIT
"""Turn VIPSeg videos into environment clips with scorer annotations.

VIPSeg (Miao et al., CVPR 2022; non-commercial research use) publishes per-frame panoptic masks
with instance IDs that persist across a video's frames. The masks are checked by people, not
produced by this project's models, so they can stand in for ground truth. This module reads
the extracted dataset and writes, for one video:

- ``<name>.mp4``: the annotated frames, in order, at a constant rate;
- ``<name>.json`` plus ``masks/*.png``: a semantic-scope annotation in the scorer's format;
- ``<name>.build-report.json``: the measurements behind every tag and the mapping used.

Mask encoding (read off the dataset's own ``create_panoptic_video_labels.py`` and confirmed on
its masks): ``0`` is unlabelled; a value below 125 is a stuff class with ``id = value - 1``; a
value of 125 or more is a thing, ``id = value // 100 - 1`` and ``instance = value % 100``.

Mapping to the scorer:

- ``wall``, ``ceiling`` and ``floor`` become surfaces. A class absent from a well-labelled frame
  is a class-negative frame for it.
- Vehicle things the scan can name become ``vehicle`` instances; things its vocabulary can name
  (see ``ALIASES``) become ``object`` instances. Instance IDs are ``<class>-<instance>`` and stay
  stable within a video.
- Unlabelled pixels, people, animals, things the vocabulary cannot name (doors and windows are
  surfaces to it) and stuff classes that a detector may legitimately report as an object become
  ignore regions, so they are neither detections nor misses. Terrain, buildings,
  sky and water stay scored as background: a predicted ceiling over sky still counts.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_SCHEMA = "skeleton-maker.environment-annotations/1"
CATEGORY_FILE = Path("VIPSeg_720P") / "panoVIPSeg_categories.json"
STRUCTURAL = {"wall": 0, "ceiling": 1, "floor": 13}
#: Vehicle classes the scan's vocabulary names (car, bus, truck, bicycle, motorcycle).
VEHICLE_IDS = frozenset({48, 49, 50, 51, 52})
PERSON_ID = 60
#: Model label -> VIPSeg thing class (lower case), chosen by Jev on 2026-10-04 before any model
#: ran on the corpus; ``bucket`` (0.56) and ``monitor`` (0.64) are its least certain choices.
#: Labels with no VIPSeg thing class (shelf, cabinet, knife, hammer, drill, lamp, book) are
#: absent here, so their predictions are scored only where a mask is not already ignored.
ALIASES = {
    "chair": "chair_or_seat",
    "stool": "chair_or_seat",
    "table": "table_or_desk",
    "desk": "table_or_desk",
    "bottle": "bottle_or_cup",
    "cup": "bottle_or_cup",
    "bowl": "tub_or_bowl_or_pot",
    "pot": "tub_or_bowl_or_pot",
    "bucket": "tub_or_bowl_or_pot",
    "bag": "bag_or_package",
    "backpack": "bag_or_package",
    "laptop": "computer",
    "phone": "mobile_phone",
    "monitor": "screen_or_television",
}
#: VIPSeg thing classes the vocabulary can name; every other thing is an ignore region.
SCORED_OBJECT_CLASSES = frozenset(
    {*ALIASES.values(), "sofa", "bed", "bench", "plate", "box", "ladder"}
)
#: Animals: dynamic beings, ignored like people.
ANIMAL_IDS = frozenset({61, 62, 63, 64, 65})
#: Stuff classes scored as background (Jev's classification of the 62 stuff classes, 2026-10-04).
BACKGROUND_STUFF = frozenset(
    {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 32, 33, 35, 36, 37, 38, 39}
)
POLICY = {
    "background_stuff_ids": sorted(BACKGROUND_STUFF),
    "low_confidence_ignored": [
        "stair",
        "windmill",
        "well_or_well_lid",
        "other_construction",
        "wood",
    ],
    "low_confidence_background": ["grandstand"],
}
FRAME_RATE = 5
INTERVAL_S = 2.0
#: A cut kept in a window needs this many frames on each side, or the shot detector misses it.
CUT_MARGIN = 3
#: A frame is "well labelled" for a class-negative claim when under this share is unlabelled.
MAX_VOID_FRACTION = 0.05
MIN_NEGATIVE_FRACTION = 0.0
CELL = 6
#: Tag thresholds: stated here so a reader can see exactly what each tag measured.
TRIPOD_SHIFT_PX = 0.3
VEHICLE_MOVING_FRACTION = 0.015
VEHICLE_STATIONARY_FRACTION = 0.003
OCCLUDER_MIN_AREA = 0.002
INDOOR_CEILING_FRACTION = 0.02
#: Share of pixels in terrain, building, sky or water classes that makes a clip outdoor.
OUTDOOR_MIN = 0.15
#: A clip with a wall and a floor in half its frames is indoor below this outdoor share.
INDOOR_OUTDOOR_MAX = 0.02


class ConversionError(RuntimeError):
    """The video cannot be converted."""


def load_categories(root: Path) -> dict[int, dict]:
    path = root / CATEGORY_FILE
    return {c["id"]: c for c in json.loads(path.read_text())}


def decode(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Category id (``-1`` for unlabelled) and instance number (``-1`` for stuff) per pixel."""
    value = mask.astype(np.int64)
    thing = value >= 125
    category = np.where(thing, value // 100, value) - 1
    instance = np.where(thing, value % 100, -1)
    return category, instance


def family_of(category: int, cats: dict[int, dict]) -> str | None:
    """``vehicle``, ``object`` or ``None`` when the class is not scored as an instance."""
    if not cats[category]["isthing"]:
        return None
    if category in VEHICLE_IDS:
        return "vehicle"
    return "object" if cats[category]["name"].lower() in SCORED_OBJECT_CLASSES else None


def is_ignored(category: int, cats: dict[int, dict]) -> bool:
    if cats[category]["isthing"]:
        return family_of(category, cats) is None
    return category not in BACKGROUND_STUFF and category not in STRUCTURAL.values()


def frame_labels(mask: np.ndarray, cats: dict[int, dict]) -> dict:
    category, instance = decode(mask)
    void = category < 0
    out: dict = {
        "void": void,
        "surfaces": {},
        "instances": {},
        "ignore": void.copy(),
        "person": None,
    }
    for name, ident in STRUCTURAL.items():
        region = category == ident
        if region.any():
            out["surfaces"][name] = region
    for ident in np.unique(category[category >= 0]):
        ident = int(ident)
        if is_ignored(ident, cats):
            out["ignore"] |= category == ident
        family = family_of(ident, cats)
        if family is not None:
            for number in np.unique(instance[(category == ident) & (instance >= 0)]):
                out["instances"][(cats[ident]["name"], int(number))] = (
                    family,
                    (category == ident) & (instance == number),
                )
    out["person"] = category == PERSON_ID
    return out


def class_stats(root: Path, video: str, cats: dict[int, dict]) -> dict:
    """Mask-only statistics used to shortlist videos; no image is read."""
    masks = sorted((root / "panomasks" / video).glob("*.png"))
    seen_surface = dict.fromkeys(STRUCTURAL, 0)
    vehicles: dict[str, set[int]] = {}
    objects: set[str] = set()
    person = frames = 0
    void = 0.0
    shape = None
    numbers = []
    for path in masks:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ConversionError(f"{path}: unreadable mask")
        shape = mask.shape
        numbers.append(int(path.stem))
        frames += 1
        values = np.unique(mask)
        void += float((mask == 0).mean())
        for value in (int(v) for v in values if v):
            ident = (value // 100 if value >= 125 else value) - 1
            for name, structural in STRUCTURAL.items():
                if ident == structural and value < 125:
                    seen_surface[name] += 1
            if value < 125:
                continue
            family = family_of(ident, cats)
            if family == "vehicle":
                vehicles.setdefault(cats[ident]["name"], set()).add(value % 100)
            elif family == "object":
                objects.add(cats[ident]["name"])
        person += int(any(v >= 125 and v // 100 - 1 == PERSON_ID for v in values))
    strides = sorted({b - a for a, b in pairwise(numbers)})
    return {
        "video": video,
        "frames": frames,
        "height": shape[0] if shape else 0,
        "width": shape[1] if shape else 0,
        "strides": strides,
        "even_size": bool(shape and shape[0] % 2 == 0 and shape[1] % 2 == 0),
        "surface_frames": seen_surface,
        "vehicle_instances": {k: len(v) for k, v in sorted(vehicles.items())},
        "object_classes": sorted(objects),
        "person_frames": person,
        "mean_void_fraction": void / frames if frames else 1.0,
    }


def _write_png(path: Path, region: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), region.astype(np.uint8) * 255):
        raise ConversionError(f"cannot write {path}")
    return path.name


def encode_clip(
    images: list[Path], destination: Path, frame_rate: int, size: tuple[int, int] | None = None
) -> str:
    if shutil.which("ffmpeg") is None:
        raise ConversionError("ffmpeg is required to encode the clip")
    with tempfile.TemporaryDirectory() as tmp:
        for index, image in enumerate(images):
            (Path(tmp) / f"{index:06d}.jpg").symlink_to(image.resolve())
        scale = ["-vf", f"scale={size[0]}:{size[1]}:flags=area"] if size else []
        command = [
            "ffmpeg", "-loglevel", "error", "-y", "-framerate", str(frame_rate),
            "-i", str(Path(tmp) / "%06d.jpg"), *scale, "-c:v", "libx264", "-crf", "12",
            "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart", str(destination),
        ]  # fmt: skip
        subprocess.run(command, check=True)  # noqa: S603 - fixed argument list, no shell
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def target_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """The encoded size: unchanged, or ``max_width`` wide with an even height of the same aspect."""
    if max_width is None or width <= max_width:
        return width, height
    return max_width, 2 * round(max_width * height / width / 2)


def choose_window(
    total: int, shots: list[tuple[int, int]], rate: int, max_frames: int | None
) -> int:
    """First annotated frame of the window kept: a cut stays inside it when the video has one."""
    if max_frames is None or total <= max_frames:
        return 0
    cuts = [first for first, _ in shots[1:]]
    need = int(INTERVAL_S * rate)

    def holds_interval(start: int) -> bool:
        end = start + max_frames - 1
        return any(
            min(last, end) - max(first, start) >= need for first, last in shots if first <= end
        )

    starts = range(total - max_frames + 1)
    if cuts:
        for start in starts:
            inside = any(
                start + CUT_MARGIN <= cut <= start + max_frames - 1 - (CUT_MARGIN - 1)
                for cut in cuts
            )
            if inside and holds_interval(start):
                return start
    for start in starts:
        if holds_interval(start):
            return start
    raise ConversionError(f"no {max_frames}-frame window holds a two-second run inside one shot")


def background_shift(images: list[Path]) -> float:
    """Median frame-to-frame translation in pixels at 320 px wide; near zero for a tripod."""
    shifts = []
    previous = None
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ConversionError(f"{path}: unreadable image")
        scale = 320 / gray.shape[1]
        small = cv2.resize(gray, (320, max(2, round(gray.shape[0] * scale)))).astype(np.float32)
        if previous is not None:
            (dx, dy), _ = cv2.phaseCorrelate(previous, small)
            shifts.append(float(np.hypot(dx, dy)))
        previous = small
    return float(np.median(shifts)) if shifts else 0.0


def _components(region: np.ndarray) -> int:
    return int(cv2.connectedComponents(region.astype(np.uint8))[0]) - 1


def measure_tags(
    frames: list[dict], width: int, shift: float, shots: int
) -> tuple[list[str], dict]:
    """Tags from labels and measurements; a tag is absent when it was not measured."""
    area = frames[0]["void"].size
    ceiling = [f["surfaces"]["ceiling"].mean() for f in frames if "ceiling" in f["surfaces"]]
    sky = float(np.mean([f.get("sky", 0.0) for f in frames]))
    outdoor = float(np.mean([f.get("outdoor", 0.0) for f in frames]))
    enclosed = (
        outdoor < INDOOR_OUTDOOR_MAX
        and np.mean(["wall" in f["surfaces"] for f in frames]) >= 0.5
        and np.mean(["floor" in f["surfaces"] for f in frames]) >= 0.5
    )
    if sum(1 for c in ceiling if c >= INDOOR_CEILING_FRACTION) >= 2:
        tags = ["indoor"]
    elif outdoor >= OUTDOOR_MIN:
        tags = ["outdoor"]
    else:
        tags = ["indoor"] if enclosed else []
    if not any(f["person"].any() for f in frames):
        tags.append("person_free")
    if shots > 1:
        tags.append("edited_cut")
    occluded = handled = False
    tracks: dict[tuple, list[tuple[float, float]]] = {}
    for f in frames:
        for key, (family, region) in f["instances"].items():
            if int(region.sum()) >= OCCLUDER_MIN_AREA * area and _components(region) >= 2:
                occluded = True
            if family == "object" and f["person"].any():
                near = cv2.dilate(f["person"].astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                if (region & near).any() and region.sum() < f["person"].sum():
                    handled = True
            if family == "vehicle":
                ys, xs = np.nonzero(region)
                tracks.setdefault(key, []).append((float(xs.mean()), float(ys.mean())))
    if occluded:
        tags.append("occlusion")
    if handled:
        tags.append("handled_object")
    fixed = shift < TRIPOD_SHIFT_PX
    moving = stationary = False
    if fixed:
        tags += ["tripod", "underconstrained"]
        for points in tracks.values():
            steps = [np.hypot(b[0] - a[0], b[1] - a[1]) / width for a, b in pairwise(points)]
            if len(steps) >= 2:
                moving |= max(steps) >= VEHICLE_MOVING_FRACTION
                stationary |= max(steps) <= VEHICLE_STATIONARY_FRACTION
        if moving:
            tags.append("vehicle_moving")
        if stationary:
            tags.append("vehicle_stationary")
    measured = {
        "ceiling_fraction_by_frame": [round(float(c), 4) for c in ceiling],
        "mean_sky_fraction": round(sky, 4),
        "mean_outdoor_fraction": round(outdoor, 4),
        "background_shift_px_320": round(shift, 3),
        "shots": shots,
        "occlusion_seen": occluded,
        "handled_object_seen": handled,
        "thresholds": {
            "tripod_shift_px": TRIPOD_SHIFT_PX,
            "vehicle_moving_fraction": VEHICLE_MOVING_FRACTION,
            "vehicle_stationary_fraction": VEHICLE_STATIONARY_FRACTION,
            "indoor_ceiling_fraction": INDOOR_CEILING_FRACTION,
            "outdoor_min": OUTDOOR_MIN,
            "indoor_outdoor_max": INDOOR_OUTDOOR_MAX,
        },
        "not_measured": ["motion_blur", "translating", "pan"],
    }
    return sorted(set(tags)), measured


def longest_interval(ids: list[int], shots: list[tuple[int, int]], rate: int) -> tuple[int, int]:
    """The longest run of annotated frames inside one shot that spans at least two seconds."""
    best: tuple[int, int] | None = None
    for first, last in shots:
        inside = [i for i in ids if first <= i <= last]
        if not inside or inside[-1] - inside[0] < INTERVAL_S * rate:
            continue
        if best is None or inside[-1] - inside[0] > best[1] - best[0]:
            best = (inside[0], inside[-1])
    if best is None:
        raise ConversionError("no shot holds a two-second run of annotated frames")
    return best


def build(
    root: Path,
    video: str,
    out_dir: Path,
    *,
    name: str,
    split: str,
    aliases: dict[str, str] | None = None,
    frame_rate: int = FRAME_RATE,
    encode: bool = True,
    clip_sha256: str | None = None,
    shots_override: list[tuple[int, int]] | None = None,
    shift_override: float | None = None,
    max_width: int | None = None,
    max_frames: int | None = None,
) -> dict:
    cats = load_categories(root)
    images = sorted((root / "imgs" / video).glob("*.jpg"))
    masks = sorted((root / "panomasks" / video).glob("*.png"))
    if not images or [p.stem for p in images] != [p.stem for p in masks]:
        raise ConversionError(f"{video}: images and masks do not pair up one to one")
    numbers = [int(p.stem) for p in images]
    if len({b - a for a, b in pairwise(numbers)}) > 1:
        raise ConversionError(f"{video}: annotated frames are not evenly spaced")
    first = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
    height, width = first.shape
    if height % 2 or width % 2:
        raise ConversionError(f"{video}: odd frame size {width}x{height} cannot be encoded")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_width, out_height = target_size(width, height, max_width)
    if encode:
        from skeleton_maker import shots as shot_module

        def detect(path: Path, count: int) -> list[tuple[int, int]]:
            cuts = [c["frame_id"] for c in shot_module.detect_cuts(str(path), count, range(count))]
            starts = [0, *cuts]
            return [(a, b - 1) for a, b in zip(starts, [*cuts, count], strict=True)]

        if max_frames is not None and len(images) > max_frames:
            with tempfile.TemporaryDirectory() as probe:
                encode_clip(images, Path(probe) / "full.mp4", frame_rate, (320, 180))
                full_shots = detect(Path(probe) / "full.mp4", len(images))
            first_kept = choose_window(len(images), full_shots, frame_rate, max_frames)
            images = images[first_kept : first_kept + max_frames]
            masks = masks[first_kept : first_kept + max_frames]
            numbers = numbers[first_kept : first_kept + max_frames]
        sha = encode_clip(images, out_dir / f"{name}.mp4", frame_rate, (out_width, out_height))
        shots = detect(out_dir / f"{name}.mp4", len(images))
    else:
        if clip_sha256 is None or shots_override is None:
            raise ConversionError("clip_sha256 and shots_override are required without encoding")
        sha, shots = clip_sha256, shots_override
        first_kept = choose_window(len(images), shots, frame_rate, max_frames)
        if max_frames is not None:
            end = first_kept + max_frames - 1
            shots = [
                (max(a, first_kept) - first_kept, min(b, end) - first_kept)
                for a, b in shots
                if a <= end and b >= first_kept
            ]
            images = images[first_kept : first_kept + max_frames]
            masks = masks[first_kept : first_kept + max_frames]
            numbers = numbers[first_kept : first_kept + max_frames]
    labelled = []
    records = []
    for index, path in enumerate(masks):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (out_width, out_height) != (width, height):
            mask = cv2.resize(mask, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
        labels = frame_labels(mask, cats)
        category, _ = decode(mask)
        labels["sky"] = float((category == 28).mean())
        labels["outdoor"] = float(np.isin(category, list(BACKGROUND_STUFF)).mean())
        labelled.append(labels)
        tag = f"{name}-{index:04d}"
        surfaces = [
            {
                "class": cls,
                "mask": f"masks/{_write_png(out_dir / 'masks' / f'{tag}-{cls}.png', region)}",
            }
            for cls, region in labels["surfaces"].items()
        ]
        well_labelled = float(labels["void"].mean()) <= MAX_VOID_FRACTION
        negatives = [c for c in STRUCTURAL if c not in labels["surfaces"]] if well_labelled else []
        instances = []
        for (cls, number), (family, region) in labels["instances"].items():
            mask_name = _write_png(out_dir / "masks" / f"{tag}-{cls}-{number}.png", region)
            instances.append(
                {
                    "id": f"{cls}-{number}",
                    "family": family,
                    "class": cls,
                    "visibility": "visible",
                    "mask": f"masks/{mask_name}",
                }
            )
        record = {
            "frame_id": index,
            "surfaces": surfaces,
            "negative_classes": negatives,
            "instances": instances,
        }
        if labels["ignore"].any():
            ignore_name = _write_png(out_dir / "masks" / f"{tag}-ignore.png", labels["ignore"])
            record["ignore"] = [{"mask": f"masks/{ignore_name}"}]
        records.append(record)
    shift = shift_override if shift_override is not None else background_shift(images)
    tags, measured = measure_tags(labelled, out_width, shift, len(shots))
    interval = longest_interval(list(range(len(images))), shots, frame_rate)
    annotation = {
        "schema": ANNOTATION_SCHEMA,
        "clip": name,
        "source_sha256": sha,
        "split": split,
        "scope": "semantic",
        "width": out_width,
        "height": out_height,
        "frame_rate": [frame_rate, 1],
        "tags": tags,
        "aliases": ALIASES if aliases is None else aliases,
        "provenance": {
            "kind": "published_dataset",
            "source": f"VIPSeg video {video}",
            "citation": "Miao, Wang, Wu, Yang et al., Large-scale Video Panoptic Segmentation in the Wild, CVPR 2022",
            "license": "non-commercial research purpose only",
            "quality_control": (
                "four expert annotators double-check machine-propagated masks, which are refined "
                "repeatedly; instance identities come from a tracker with human correction"
            ),
        },
        "shots": [{"first_frame": a, "last_frame": b} for a, b in shots],
        "frames": records,
        "tracking_intervals": [{"first_frame": interval[0], "last_frame": interval[1]}],
    }
    report = {
        "video": video,
        "source_frames": numbers,
        "source_size": [width, height],
        "encoded_size": [out_width, out_height],
        "clip_sha256": sha,
        "frame_rate": frame_rate,
        "policy": POLICY,
        "measured": measured,
        "tracking_interval": list(interval),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(annotation, indent=2) + "\n")
    (out_dir / f"{name}.build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"annotation": annotation, "report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stats = sub.add_parser("stats", help="mask-only statistics for a list of videos")
    stats.add_argument("root", type=Path)
    stats.add_argument("videos", type=Path, help="a split file such as val.txt")
    stats.add_argument("out", type=Path)
    convert = sub.add_parser("convert", help="convert one video")
    convert.add_argument("root", type=Path)
    convert.add_argument("video")
    convert.add_argument("out_dir", type=Path)
    convert.add_argument("--name", required=True)
    convert.add_argument("--split", choices=("development", "heldout"), required=True)
    convert.add_argument("--aliases", type=Path, help="JSON object: model label -> VIPSeg class")
    convert.add_argument("--max-width", type=int, help="scale frames and masks down to this width")
    convert.add_argument("--max-frames", type=int, help="keep at most this many annotated frames")
    args = parser.parse_args(argv)
    try:
        if args.command == "stats":
            cats = load_categories(args.root)
            rows = []
            for video in args.videos.read_text().split():
                try:
                    rows.append(class_stats(args.root, video, cats))
                except ConversionError as exc:
                    rows.append({"video": video, "error": str(exc)})
            args.out.write_text(json.dumps(rows, indent=1) + "\n")
            print(f"{len(rows)} videos")
        else:
            aliases = json.loads(args.aliases.read_text()) if args.aliases else None
            result = build(
                args.root,
                args.video,
                args.out_dir,
                name=args.name,
                split=args.split,
                aliases=aliases,
                max_width=args.max_width,
                max_frames=args.max_frames,
            )
            print(f"{args.name}: tags {result['annotation']['tags']}")
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
