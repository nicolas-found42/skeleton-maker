# SPDX-License-Identifier: MIT
"""Export an offline 2D video, pose and environment inspection page."""

from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path
from urllib.parse import quote

from . import artifacts, envmanifest, poses
from .constants import NOVA77_SKELETON_LINKS
from .envmanifest import ManifestError
from .path_safety import is_same_or_ancestor, paths_overlap, same_path


def validate_viewer_paths(
    manifest_path: str | os.PathLike,
    video_path: str | os.PathLike,
    output_path: str | os.PathLike,
    pose_path: str | os.PathLike | None = None,
    calibration_path: str | os.PathLike | None = None,
) -> None:
    """Refuse viewer destinations that could replace or nest an existing input artifact."""
    manifest = Path(manifest_path).resolve()
    video = Path(video_path).resolve()
    output = Path(output_path).resolve()
    bundle = output.with_name(f"{output.stem}.viewer.assets").resolve()
    environment_assets = envmanifest.assets_dir_for(manifest).resolve()
    poses_path = Path(pose_path).resolve() if pose_path is not None else None
    calibration = Path(calibration_path).resolve() if calibration_path is not None else None

    protected_files = [("source video", video), ("environment manifest", manifest)]
    if poses_path is not None:
        protected_files.append(("pose file", poses_path))
    if calibration is not None:
        protected_files.append(("calibration file", calibration))

    for label, path in protected_files:
        if same_path(output, path):
            raise ManifestError(f"viewer output would overwrite the {label}: {path}")
        if is_same_or_ancestor(bundle, path):
            raise ManifestError(f"viewer asset bundle would contain the {label}: {path}")

    if is_same_or_ancestor(environment_assets, output):
        raise ManifestError(
            f"viewer output would overwrite an environment asset in {environment_assets}"
        )
    if paths_overlap(bundle, environment_assets):
        raise ManifestError(f"viewer asset bundle conflicts with environment assets: {bundle}")


def build_viewer_html(
    payload: dict,
    *,
    title: str,
    bundle_ref: str = "viewer.viewer.assets",
    video_name: str = "source.mp4",
) -> str:
    """Create HTML with JSON data escaped for script-data context and a local JS companion."""
    for observation in payload.get("observations", []):
        mask = observation.get("mask")
        if mask is not None:
            envmanifest.resolve_asset(Path("."), mask["asset"], "viewer mask")
    payload_text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    payload_text = payload_text.replace("<", "\\u003c").replace("&", "\\u0026")
    payload_text = payload_text.replace(">", "\\u003e")
    template = (resources.files("skeleton_maker") / "web" / "environment-viewer.html").read_text(
        encoding="utf-8"
    )
    substitutions = {
        "__TITLE__": html.escape(title, quote=True),
        "__BUNDLE__": html.escape(bundle_ref, quote=True),
        "__VIDEO__": html.escape(quote(video_name, safe="."), quote=True),
        "__PAYLOAD__": payload_text,
    }
    import re

    return re.sub("|".join(substitutions), lambda match: substitutions[match[0]], template)


def _pose_payload(path: str | None) -> list[dict]:
    if path is None:
        return []
    result = []
    for record in poses.read_records(path):
        people = []
        for detection in record["detections"]:
            people.append(
                {
                    key: detection[key]
                    for key in (
                        "tracking_id",
                        "bbox",
                        "keypoints_2d",
                        "keypoints_confidence",
                    )
                    if key in detection
                }
            )
        result.append({"frame_id": record["frame_id"], "detections": people})
    return result


def prepare_viewer(
    manifest_path: str | os.PathLike,
    video_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    staged_html: str | os.PathLike,
    staged_bundle: str | os.PathLike,
    pose_path: str | os.PathLike | None = None,
) -> None:
    """Build the viewer HTML and bundle at staging paths without publishing either."""
    manifest_path, source, output = (
        Path(manifest_path).resolve(),
        Path(video_path).resolve(),
        Path(output_path).resolve(),
    )
    staged_html, staged_bundle = Path(staged_html), Path(staged_bundle)
    if not source.is_file():
        raise ManifestError(f"viewer source video is missing: {source}")
    manifest = envmanifest.load_manifest(manifest_path)
    if artifacts.sha256_file(source) != manifest["source"]["sha256"]:
        raise ManifestError("viewer source video hash does not match manifest source")
    manifest_pose_path = manifest["poses"]["path"] if manifest["poses"] else None
    validate_viewer_paths(manifest_path, source, output, manifest_pose_path)
    pose_source = pose_path if pose_path is not None else manifest_pose_path
    bundle = output.with_name(f"{output.stem}.viewer.assets")
    data = {
        "source": manifest["source"],
        "processed_frames": manifest["processed_frames"],
        "shots": manifest["shots"],
        "entities": manifest["entities"],
        "observations": [
            {
                **observation,
                "mask": (
                    {**observation["mask"], "asset": f"environment/{observation['mask']['asset']}"}
                    if observation.get("mask") is not None
                    else None
                ),
            }
            for observation in manifest["observations"]
        ],
        "geometry": manifest["geometry"],
        "poses": _pose_payload(os.fspath(pose_source) if pose_source is not None else None),
        "skeleton_links": NOVA77_SKELETON_LINKS,
    }
    staged_html.parent.mkdir(parents=True, exist_ok=True)
    staged_bundle.mkdir(parents=True, exist_ok=True)
    video_name = f"source{source.suffix or '.mp4'}"
    shutil.copyfile(source, staged_bundle / video_name)
    js_source = resources.files("skeleton_maker") / "web" / "environment-viewer.js"
    (staged_bundle / "viewer.js").write_bytes(js_source.read_bytes())
    mask_root = envmanifest.assets_dir_for(manifest_path)
    for observation in manifest["observations"]:
        mask = observation.get("mask")
        if mask is None:
            continue
        rel = mask["asset"]
        target = envmanifest.resolve_asset(mask_root, rel, "viewer mask")
        if not target.is_file():
            raise ManifestError(f"viewer mask asset {rel!r} is missing")
        mask_target = staged_bundle / "environment" / rel
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, mask_target)
    bundle_ref = quote(bundle.name, safe="")
    page = build_viewer_html(data, title=output.stem, bundle_ref=bundle_ref, video_name=video_name)
    staged_html.write_text(page, encoding="utf-8")


def export_viewer(manifest_path: str | os.PathLike, video_path: str, output_path: str) -> None:
    """Write the viewer and its sibling ``<stem>.viewer.assets`` bundle."""
    output = Path(output_path).resolve()
    bundle = output.with_name(f"{output.stem}.viewer.assets")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.viewer-", dir=output.parent) as temp:
        staged_bundle = Path(temp) / bundle.name
        staged_page = Path(temp) / output.name
        prepare_viewer(
            manifest_path,
            video_path,
            output,
            staged_html=staged_page,
            staged_bundle=staged_bundle,
        )
        artifacts.publish(output, staged_page.read_text(encoding="utf-8"), staged_bundle, bundle)
    print(f"  viewer  : {output} (+ local companion bundle {bundle})")
