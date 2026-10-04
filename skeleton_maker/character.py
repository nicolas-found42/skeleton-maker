# SPDX-License-Identifier: MIT
"""Turn any ``pose.json`` into an animated character stage (one self-contained HTML file).

Each tracked person becomes a character built from a declarative JSON spec, driven
by the person's 3D joints. The output opens in any browser, works offline, can be
orbited, scrubbed and recorded to WebM. See ``characters/*.json`` for the specs
and the README for the spec format.
"""

from __future__ import annotations

import html as htmllib
import json
import math
import os
import re
import sys
from importlib import resources
from pathlib import Path
from urllib.parse import quote

from .nova77 import CANON
from .stage import Options, build_stage, load_frames
from .utils import die

SHAPES = {"box", "cylinder", "capsule", "sphere", "cone", "icosa", "torus"}
MATERIAL_TYPES = {"standard", "basic"}


class SpecError(ValueError):
    pass


def _joint_names(value, where: str) -> list:
    names = value if isinstance(value, list) else [value]
    if not names:
        raise SpecError(f"{where}: needs at least one joint")
    for n in names:
        if n not in CANON:
            raise SpecError(f"{where}: unknown joint {n!r}. Joints: {', '.join(CANON)}")
    return names


def _number(value, where: str, *, positive=False, unit=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecError(f"{where}: needs a finite number")
    if positive and value <= 0:
        raise SpecError(f"{where}: must be positive")
    if unit and not 0 <= value <= 1:
        raise SpecError(f"{where}: must be between 0 and 1")


def _vector(value, where: str, length=3, *, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise SpecError(f"{where}: needs a list of {length} numbers")
    for i, item in enumerate(value):
        _number(item, f"{where}[{i}]", positive=positive)


def _count(value, where: str, maximum: int):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise SpecError(f"{where}: needs an integer between 1 and {maximum}")


def validate_spec(spec: dict) -> dict:
    """Check a character spec and return it; raises :class:`SpecError` with a pointed message."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")
    name = spec.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        raise SpecError("spec needs a 'name' that is a plain identifier, e.g. 'my_robot'")
    materials = spec.get("materials")
    if not isinstance(materials, dict) or not materials:
        raise SpecError(
            f"{name}: 'materials' must be an object mapping a material name to {{type: standard|basic, ...}}"
        )
    for m, d in materials.items():
        where = f"{name}.materials[{m!r}]"
        if not isinstance(d, dict):
            raise SpecError(f"{where}: material must be an object")
        if not isinstance(d.get("type"), str) or d["type"] not in MATERIAL_TYPES:
            raise SpecError(f"{name}: material {m!r} needs type 'standard' or 'basic'")
        for key in ("roughness", "metalness", "opacity"):
            if key in d:
                _number(d[key], f"{where}.{key}", unit=True)
        if "wireframe" in d and not isinstance(d["wireframe"], bool):
            raise SpecError(f"{where}.wireframe: needs a boolean")
    palettes = spec.get("palettes")
    if not isinstance(palettes, list) or not palettes:
        raise SpecError(
            f"{name}: 'palettes' must be a non-empty list of {{material: '#rrggbb'}} objects"
        )
    for i, pal in enumerate(palettes):
        if not isinstance(pal, dict):
            raise SpecError(f"{name}: palette {i} must be an object")
        for m in materials:
            if m not in pal:
                raise SpecError(f"{name}: palette {i} is missing a colour for material {m!r}")
            if not isinstance(pal[m], str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", pal[m]):
                raise SpecError(f"{name}: palette {i} material {m!r} needs a '#rrggbb' colour")
    parts = spec.get("parts")
    if not isinstance(parts, list) or not parts:
        raise SpecError(f"{name}: 'parts' must be a non-empty list")
    for i, p in enumerate(parts):
        where = f"{name}.parts[{i}]"
        if not isinstance(p, dict):
            raise SpecError(f"{where}: part must be an object")
        kind = p.get("type")
        if kind == "limb":
            frm, to = (
                _joint_names(p.get("from"), where + ".from"),
                _joint_names(p.get("to"), where + ".to"),
            )
            if len(frm) > 1 and len(to) > 1 and len(frm) != len(to):
                raise SpecError(
                    f"{where}: 'from' and 'to' lists must be the same length (or one of them a single joint)"
                )
            if p.get("shape", "capsule") not in ("box", "cylinder", "capsule"):
                raise SpecError(f"{where}: limb shape must be box, cylinder or capsule")
            if not (("r" in p) or ("w" in p and "d" in p)):
                raise SpecError(
                    f"{where}: a limb needs 'r' (radius) or both 'w' and 'd' (full widths, metres at 1.75 m height)"
                )
            for key in ("r", "r2", "w", "d"):
                if key in p:
                    _number(p[key], f"{where}.{key}", positive=True)
            if "extend" in p:
                _vector(p["extend"], f"{where}.extend", 2)
        elif kind == "prop":
            _joint_names(p.get("joint"), where + ".joint")
            if (
                not isinstance(p.get("shape", "sphere"), str)
                or p.get("shape", "sphere") not in SHAPES
            ):
                raise SpecError(f"{where}: prop shape must be one of {sorted(SHAPES)}")
            if p.get("frame", "body") not in ("body", "head"):
                raise SpecError(f"{where}: frame must be 'body' or 'head'")
            for key in ("size", "pos", "rot"):
                if key in p:
                    _vector(p[key], f"{where}.{key}", positive=key == "size")
        elif kind == "chain":
            if not isinstance(p.get("joint"), str):
                raise SpecError(f"{where}.joint: needs a single joint name")
            _joint_names(p.get("joint"), where + ".joint")
            for key in ("dir", "n", "len", "r", "mat", "lag"):
                if key not in p:
                    raise SpecError(f"{where}: a chain needs {key!r}")
            _vector(p["dir"], f"{where}.dir")
            if not any(p["dir"]):
                raise SpecError(f"{where}.dir: must not be zero")
            _count(p["n"], f"{where}.n", 256)
            _number(p["len"], f"{where}.len", positive=True)
            _vector(p["r"], f"{where}.r", 2, positive=True)
            _number(p["lag"], f"{where}.lag", unit=True)
            if "tip" in p and (not isinstance(p["tip"], str) or p["tip"] not in materials):
                raise SpecError(f"{where}.tip: unknown material")
        else:
            raise SpecError(f"{where}: type must be limb, prop or chain, not {kind!r}")
        if not isinstance(p.get("mat"), str) or p["mat"] not in materials:
            raise SpecError(
                f"{where}: unknown material {p.get('mat')!r}; defined: {sorted(materials)}"
            )
    trails = spec.get("trails")
    if trails is not None:
        if not isinstance(trails, dict):
            raise SpecError(f"{name}.trails: must be an object")
        _joint_names(trails.get("joints"), f"{name}.trails.joints")
        if not isinstance(trails.get("joints"), list):
            raise SpecError(f"{name}.trails.joints: must be a list")
        _count(trails.get("length"), f"{name}.trails.length", 4096)
        if not isinstance(trails.get("mat"), str) or trails["mat"] not in materials:
            raise SpecError(f"{name}.trails: unknown material {trails.get('mat')!r}")
    return spec


def builtin_specs() -> dict:
    """The characters shipped with the package, validated, keyed by name."""
    out = {}
    folder = resources.files("skeleton_maker") / "characters"
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".json"):
            spec = validate_spec(json.loads(entry.read_text()))
            out[spec["name"]] = spec
    return out


def _asset(name: str) -> str:
    return (resources.files("skeleton_maker") / "web" / name).read_text()


def build_html(stage, specs: dict, options: dict, title: str) -> str:
    """Assemble the single-file viewer."""
    page = _asset("stage.html")

    def js(obj) -> str:  # Keep script-data parser control sequences out of user JSON.
        return json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c")

    replacements = {
        "__TITLE__": htmllib.escape(title),
        "__SPECS__": js(specs),
        "__OPTIONS__": js(options),
        "__PAYLOAD__": stage.payload(),
        "__THREE__": _asset("three.module.min.js"),
        "__VIEWER__": _asset("viewer.js"),
    }
    # Substitute the original template once; inserted user text is never a new template slot.
    return re.sub("|".join(replacements), lambda match: replacements[match[0]], page)


def make_stage_html(
    pose_json: str,
    out: str,
    *,
    character: str = "auto",
    extra_specs: list | None = None,
    video: str | None = None,
    fps: float = 30.0,
    fov: float = 45.0,
    min_conf: float = 0.0,
    scale: float = 1.0,
    title: str | None = None,
) -> dict:
    """Write the stage HTML; returns a summary dict."""
    _number(fps, "fps", positive=True)
    _number(scale, "scale", positive=True)
    _number(fov, "fov", positive=True)
    if fov >= 180:
        raise SpecError("fov: must be less than 180 degrees")
    _number(min_conf, "min_conf", unit=True)
    specs = builtin_specs()
    for path in extra_specs or []:
        try:
            with open(path) as fh:
                spec = validate_spec(json.load(fh))
        except OSError as exc:
            raise SpecError(f"cannot read spec {path}: {exc.strerror}") from exc
        except json.JSONDecodeError as exc:
            raise SpecError(f"{path}: {exc.msg} at line {exc.lineno}") from exc
        specs[spec["name"]] = spec
    if character != "auto" and character not in specs:
        die(f"unknown character {character!r}. Available: {', '.join(sorted(specs))}")
    frames = load_frames(pose_json)
    if not frames:
        die(f"{pose_json} has no frames")
    stage = build_stage(frames, Options(fps=fps, min_conf=min_conf))
    shots = stage.meta["shots"]
    if not shots:
        die("no usable bodies found in the pose file (no frame had hips, chest and head)")
    options = {
        "character": character,
        "fov": fov,
        "scale": scale,
        "title": title or os.path.basename(pose_json),
    }
    if video:
        video_path = Path(video).resolve()
        try:
            relative = os.path.relpath(video_path, Path(out).resolve().parent)
        except ValueError:  # Windows paths on different drives have no relative form.
            options["video"] = video_path.as_uri()
        else:
            options["video"] = quote(Path(relative).as_posix(), safe="/")
    html = build_html(stage, specs, options, options["title"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(html)
    people = sum(len(s["tracks"]) for s in shots)
    return {
        "shots": len(shots),
        "tracks": people,
        "frames": sum(s["n"] for s in shots),
        "bytes": len(html),
        "characters": sorted(specs),
    }


def add_cli(subparsers) -> None:
    p = subparsers.add_parser(
        "character", help="turn a pose file into an animated character stage (HTML)"
    )
    p.add_argument("pose_json", nargs="?", help="pose output from `pose`")
    p.add_argument("--out", help="HTML file to write (default: <pose>.stage.html)")
    p.add_argument(
        "--character",
        default="auto",
        help="a character name, or 'auto' to cast a different one per person",
    )
    p.add_argument(
        "--spec",
        action="append",
        default=[],
        metavar="FILE",
        help="add a custom character spec (JSON); repeatable",
    )
    p.add_argument("--video", help="the source clip, shown picture-in-picture and kept in sync")
    p.add_argument("--fps", type=float, default=30.0, help="frame rate of the clip (default: 30)")
    p.add_argument(
        "--fov",
        type=float,
        default=45.0,
        help="vertical field of view of the original camera, degrees",
    )
    p.add_argument(
        "--min-conf", type=float, default=0.0, help="drop joints at or below this confidence"
    )
    p.add_argument(
        "--scale", type=float, default=1.0, help="scale every character (1.0 = fits the body)"
    )
    p.add_argument("--title", help="title shown in the viewer")
    p.add_argument("--list", action="store_true", help="list the built-in characters and exit")
    p.set_defaults(func=run_cli)


def run_cli(args) -> int:
    if args.list:
        for name, spec in builtin_specs().items():
            print(f"{name:10} {spec.get('description', '')}")
        return 0
    if not args.pose_json:
        die("give a pose.json (or --list)")
    out = args.out or os.path.splitext(args.pose_json)[0] + ".stage.html"
    try:
        info = make_stage_html(
            args.pose_json,
            out,
            character=args.character,
            extra_specs=args.spec,
            video=args.video,
            fps=args.fps,
            fov=args.fov,
            min_conf=args.min_conf,
            scale=args.scale,
            title=args.title,
        )
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {out} ({info['bytes'] / 1e6:.1f} MB): {info['tracks']} track segments in {info['shots']} shot(s), {info['frames']} frames"
    )
    print(f"open it in a browser; characters: {', '.join(info['characters'])}")
    return 0
