# SPDX-License-Identifier: MIT
"""The environment label vocabulary: the default preset, user extensions and containment.

The preset is a starting vocabulary, not a promise that every such thing in a clip is found.
People are deliberately absent: the skeletons own them.
"""

from typing import NamedTuple

FAMILIES = ("surface", "object", "vehicle")


class Entry(NamedTuple):
    label: str
    family: str


def _entries(family: str, *labels: str) -> list[Entry]:
    return [Entry(label, family) for label in labels]


PRESET: list[Entry] = [
    # structural surfaces and elements: semantic regions with visible extent
    *_entries("surface", "wall", "floor", "ceiling", "ground", "road", "stairs", "door", "window"),
    # furniture
    *_entries(
        "object", "chair", "table", "sofa", "bed", "desk", "shelf", "cabinet", "bench", "stool"
    ),
    # tools, containers and handheld objects
    *_entries(
        "object",
        "bottle",
        "cup",
        "bowl",
        "plate",
        "knife",
        "bag",
        "backpack",
        "box",
        "bucket",
        "ladder",
        "hammer",
        "drill",
        "laptop",
        "phone",
        "monitor",
        "lamp",
        "book",
        "pot",
    ),
    *_entries("vehicle", "car", "truck", "bus", "bicycle", "motorcycle"),
]

#: Model phrases that name the same thing as a vocabulary label.
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

#: Surfaces hard enough that they are prompted one at a time rather than in a group.
PROMPTED_ALONE = ("wall", "floor", "ceiling")

#: Structural elements that sit inside a larger surface.
CONTAINED = ("door", "window")
CONTAINER = "wall"
CONTAINMENT_FRACTION = 0.9


def parse_classes(text: str | None) -> list[Entry]:
    """Parse ``--classes "a, b=vehicle"`` into entries; raises ValueError with a pointed message."""
    if text is None:
        return []
    entries = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            raise ValueError("--classes contains an empty label (check for a stray comma)")
        label, sep, family = item.partition("=")
        label, family = label.strip(), (family.strip() if sep else "object")
        if "=" in family:
            raise ValueError(f"--classes {raw.strip()!r}: give one family at most, as label=family")
        if not label:
            raise ValueError(f"--classes {raw.strip()!r} has no label before '='")
        if label == "person":
            raise ValueError("'person' is not an environment label: the skeletons describe people")
        if family not in FAMILIES:
            raise ValueError(
                f"--classes {raw.strip()!r}: family {family!r} is not one of {', '.join(FAMILIES)}"
            )
        entries.append(Entry(label, family))
    return entries


def build_vocabulary(extra: list[Entry]) -> list[dict]:
    """The preset plus the user's labels; a user label that is already in the preset is not added twice."""
    vocabulary = [{"label": e.label, "family": e.family, "source": "preset"} for e in PRESET]
    known = {e.label for e in PRESET}
    for entry in extra:
        if entry.label not in known:
            known.add(entry.label)
            vocabulary.append({"label": entry.label, "family": entry.family, "source": "user"})
    return vocabulary


def _inside_fraction(child: list, parent: list) -> float:
    cx0, cy0, cx1, cy1 = child
    px0, py0, px1, py1 = parent
    area = max(cx1 - cx0, 0) * max(cy1 - cy0, 0)
    if area == 0:
        return 0.0
    w = max(min(cx1, px1) - max(cx0, px0), 0)
    h = max(min(cy1, py1) - max(cy0, py0), 0)
    return (w * h) / area


def containment(entities: list[dict], observations: list[dict]) -> list[dict]:
    """``contained_in`` relations: a door or window whose box lies inside a wall's, per frame.

    Both entities stay as they are and neither mask is touched; the relation is evidence
    from boxes only, and is reported for the frames in which it holds.
    """
    by_id = {e["id"]: e for e in entities}
    per_frame: dict[tuple, dict] = {}
    for obs in observations:
        ent = by_id[obs["entity"]]
        if obs["visibility"] != "visible" or ent["labels"]["status"] != "matched":
            continue
        per_frame.setdefault((ent["shot"], obs["frame_id"]), {}).setdefault(
            ent["labels"]["normalized"], []
        ).append((ent["id"], obs["bbox"]))
    found: dict[tuple, list[int]] = {}
    for (_shot, frame), groups in per_frame.items():
        walls = groups.get(CONTAINER, [])
        for label in CONTAINED:
            for child_id, child_box in groups.get(label, []):
                for wall_id, wall_box in walls:
                    if _inside_fraction(child_box, wall_box) >= CONTAINMENT_FRACTION:
                        found.setdefault((child_id, wall_id), []).append(frame)
    return [
        {
            "type": "contained_in",
            "child": child,
            "parent": parent,
            "frames": sorted(set(frames)),
            "evidence": "bbox",
        }
        for (child, parent), frames in sorted(found.items())
    ]
