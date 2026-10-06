# SPDX-License-Identifier: MIT
"""``skeleton-maker`` command line: scan -> clip -> track -> pose -> render.

Each stage is also usable on its own; ``all`` chains them for a local video.
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

from . import (
    __version__,
    artifacts,
    character,
    detection,
    environment,
    envmanifest,
    envscore,
    nim,
    render,
    verify,
)
from .constants import CONFORM_ENCODE_ARGS
from .utils import die, is_streamable_mp4, probe_video, require_tool, run_ffmpeg


def cmd_clip(args) -> int:
    """Cut a window out of a video and make it NIM-conformant."""
    if not os.path.isfile(args.video):
        die(f"no such file: {args.video}")
    require_tool("ffmpeg", "Install it from https://ffmpeg.org/download.html")

    src = probe_video(args.video)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ff_args = []
    if args.start:
        ff_args += ["-ss", str(args.start)]
    ff_args += ["-i", args.video]
    if args.duration:
        ff_args += ["-t", str(args.duration)]
    ff_args += [*CONFORM_ENCODE_ARGS, args.out, "-y"]
    run_ffmpeg(ff_args, f"cutting {args.out}")

    out = probe_video(args.out)
    print(
        f"source : {src['width']}x{src['height']} {src['codec']} "
        f"{src['r_fps']:.3f}fps {src['duration']:.2f}s"
    )
    print(
        f"clip   : {out['width']}x{out['height']} {out['codec']} "
        f"{out['pix_fmt']} {out['r_fps']:.3f}fps {out['duration']:.2f}s "
        f"{out['size'] / 1e6:.1f}MB -> {args.out}"
    )

    notes = []
    if out["size"] > 50_000_000:
        notes.append(
            f"the hosted endpoint caps input at 50MB; this clip is "
            f"{out['size'] / 1e6:.1f}MB. Use a shorter window, lower "
            f"resolution, or higher --crf."
        )
    if abs(out["r_fps"] - out["avg_fps"]) > 0.01:
        notes.append(
            f"frame rate is not constant ({out['r_fps']:.3f} vs "
            f"{out['avg_fps']:.3f}); the NIM rejects variable frame rate"
        )
    if not is_streamable_mp4(args.out):
        notes.append(
            "the MP4 is not streamable (moov after mdat), so the NIM will "
            "buffer the whole file before processing"
        )
    if out["pix_fmt"] != "yuv420p":
        notes.append(f"pixel format is {out['pix_fmt']}, not yuv420p")
    for note in notes:
        print(f"warning: {note}", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    return verify.cli(args)


def cmd_stubs(args) -> int:
    """Regenerate the vendored gRPC stubs from the bundled protos."""
    nim._generate_stubs()
    print(f"stubs written to {nim.GEN_DIR}")
    return 0


def cmd_all(args) -> int:
    """clip -> track -> pose -> render for a local video, in a work directory."""
    if args.environment:
        return cmd_all_environment(args)
    work = args.work or (os.path.splitext(args.video)[0] + ".skeleton")
    os.makedirs(work, exist_ok=True)
    clip = os.path.join(work, "clip.mp4")
    boxes = os.path.join(work, "boxes.txt")
    poses = os.path.join(work, "pose.json")
    overlay = args.out or os.path.join(work, "overlay.mp4")

    print(f"== work directory: {work}")
    print("== 1/4 conforming the clip")
    rc = cmd_clip(
        argparse.Namespace(video=args.video, out=clip, start=args.start, duration=args.duration)
    )
    if rc:
        return rc

    print("== 2/4 detecting and tracking people")
    rc = detection.run_track(
        argparse.Namespace(
            video=clip,
            out_bbox=boxes,
            out_json=None,
            model=args.model,
            imgsz=args.imgsz,
            conf=args.conf,
            tracker=args.tracker,
            device=args.device,
            max_bodies=args.max_bodies,
        )
    )
    if rc:
        return rc

    print("== 3/4 estimating poses with the NIM")
    summary = nim.run(clip, boxes, poses, focal_length=args.focal_length, timeout=args.timeout)
    print(
        f"   {summary['frames']} frames ({summary['frames_with_bodies']} with bodies) "
        f"in {summary['seconds']}s"
    )

    print("== 4/4 rendering the overlay")
    render.render(
        clip,
        poses,
        overlay,
        draw=args.draw,
        focal_length=args.focal_length or summary["focal_length"],
    )

    print("== verifying")
    ok = verify.run(clip, boxes, poses, overlay, report=True)
    print(f"\ndone. work directory: {work}")
    return 0 if ok else 1


def cmd_all_environment(args) -> int:
    """Build and atomically publish the pose and opt-in environment pipeline."""
    work = Path(args.work or (os.path.splitext(args.video)[0] + ".skeleton")).resolve()
    source = Path(args.video).resolve()
    clip = work / "clip.mp4"
    boxes = work / "boxes.txt"
    pose_file = work / "pose.json"
    skeleton_overlay = Path(args.out or work / "overlay.mp4").resolve()
    environment_manifest = work / "environment.json"
    environment_overlay = work / "environment-overlay.mp4"
    environment_viewer = work / "environment.html"
    environment_assets = envmanifest.assets_dir_for(environment_manifest)
    viewer_bundle = environment_viewer.with_name(f"{environment_viewer.stem}.viewer.assets")

    environment_args = argparse.Namespace(
        backend=args.environment_backend,
        classes=args.environment_classes,
        device=args.environment_device,
        geometry=args.environment_geometry,
        sample_fps=args.environment_sample_fps,
    )
    calibration = getattr(args, "environment_calibration", None)
    try:
        environment.validate_artifact_paths(
            inputs=[("source video", source), ("calibration file", calibration)],
            files=[
                ("conformed clip", clip),
                ("tracking boxes", boxes),
                ("pose file", pose_file),
                ("skeleton overlay", skeleton_overlay),
                ("environment manifest", environment_manifest),
                ("environment overlay", environment_overlay),
                ("environment viewer", environment_viewer),
            ],
            directories=[
                ("environment assets", environment_assets),
                ("viewer bundle", viewer_bundle),
            ],
        )
    except environment.ManifestError as exc:
        environment._fail_options(str(exc))
    _, environment_backend, _ = environment.preflight_options(environment_args)

    stage_paths = {}
    with contextlib.ExitStack() as stack:
        for key, destination in (
            ("clip", clip),
            ("boxes", boxes),
            ("poses", pose_file),
            ("skeleton_overlay", skeleton_overlay),
            ("manifest", environment_manifest),
            ("overlay", environment_overlay),
            ("viewer", environment_viewer),
        ):
            stage_dir = stack.enter_context(artifacts.staging_dir(destination))
            stage_paths[key] = stage_dir / destination.name

        print(f"== work directory: {work}")
        print("== 1/5 conforming the clip")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = cmd_clip(
                argparse.Namespace(
                    video=str(source),
                    out=str(stage_paths["clip"]),
                    start=args.start,
                    duration=args.duration,
                )
            )
        if rc:
            return rc

        if calibration is not None:
            clip_info = environment._probe(str(stage_paths["clip"]))
            try:
                geometry_reference = environment.envgeometry.load_reference(
                    calibration,
                    width=clip_info["width"],
                    height=clip_info["height"],
                    frame_count=clip_info["frame_count"],
                )
            except environment.envgeometry.GeometryReferenceError as exc:
                environment._fail_options(str(exc))
        else:
            geometry_reference = None
        if args.environment_geometry == "required":
            environment.validate_required_reference(geometry_reference, environment_backend)

        print("== 2/5 detecting and tracking people")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = detection.run_track(
                argparse.Namespace(
                    video=str(stage_paths["clip"]),
                    out_bbox=str(stage_paths["boxes"]),
                    out_json=None,
                    model=args.model,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    tracker=args.tracker,
                    device=args.device,
                    max_bodies=args.max_bodies,
                )
            )
        if rc:
            return rc

        print("== 3/5 estimating poses with the NIM")
        summary = nim.run(
            str(stage_paths["clip"]),
            str(stage_paths["boxes"]),
            str(stage_paths["poses"]),
            focal_length=args.focal_length,
            timeout=args.timeout,
            verbose=False,
        )
        print(
            f"   {summary['frames']} frames ({summary['frames_with_bodies']} with bodies) "
            f"in {summary['seconds']}s"
        )

        print("== 4/5 rendering the skeleton overlay")
        with contextlib.redirect_stdout(io.StringIO()):
            render.render(
                str(stage_paths["clip"]),
                str(stage_paths["poses"]),
                str(stage_paths["skeleton_overlay"]),
                draw=args.draw,
                focal_length=args.focal_length or summary["focal_length"],
                verbose=False,
            )

        print("== verifying the skeleton artifacts")
        if not verify.run(
            str(stage_paths["clip"]),
            str(stage_paths["boxes"]),
            str(stage_paths["poses"]),
            str(stage_paths["skeleton_overlay"]),
            report=False,
        ):
            return 1

        print("== 5/5 scanning and rendering the environment")
        environment_args.video = str(stage_paths["clip"])
        environment_args.source_manifest_path = str(clip)
        environment_args.poses = str(stage_paths["poses"])
        environment_args.pose_manifest_path = str(pose_file)
        environment_args.out = str(stage_paths["manifest"])
        environment_args.overlay = str(stage_paths["overlay"])
        environment_args.viewer = str(stage_paths["viewer"])
        environment_args.cache_dir = args.environment_cache_dir
        environment_args.no_cache = args.environment_no_cache
        environment_args.calibration = calibration
        with contextlib.redirect_stdout(io.StringIO()):
            environment_rc = environment.run_cli(environment_args)
        if environment_rc not in (0, environment.EXIT_PARTIAL):
            return environment_rc

        stage_assets = envmanifest.assets_dir_for(stage_paths["manifest"])
        stage_viewer_bundle = stage_paths["viewer"].with_name(
            f"{stage_paths['viewer'].stem}.viewer.assets"
        )
        publish_pairs = [
            (stage_paths["clip"], clip),
            (stage_paths["boxes"], boxes),
            (stage_paths["poses"], pose_file),
            (stage_paths["skeleton_overlay"], skeleton_overlay),
            (stage_paths["manifest"], environment_manifest),
            (stage_assets if stage_assets.exists() else None, environment_assets),
            (stage_paths["overlay"], environment_overlay),
            (stage_paths["viewer"], environment_viewer),
            (stage_viewer_bundle, viewer_bundle),
        ]
        try:
            cleanup_warnings = artifacts.publish_group(publish_pairs)
        except OSError as exc:
            die(f"combined environment outputs were not published: {exc}")
        for warning in cleanup_warnings:
            print(f"warning: outputs were published, but {warning}", file=sys.stderr)

    manifest = envmanifest.load_manifest(environment_manifest)
    environment._print_summary(manifest, environment_manifest)
    print(f"  overlay : {environment_overlay}")
    print(f"  viewer  : {environment_viewer} (+ local companion bundle {viewer_bundle})")
    print(f"done. work directory: {work}")
    return environment_rc


def cmd_download(args) -> int:
    """Grab a window of a video with yt-dlp, NIM-conformant."""
    yt_dlp = require_tool("yt-dlp", "Install it: uv pip install yt-dlp")
    ffmpeg = require_tool("ffmpeg")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    section = f"*{args.start}-{args.start + args.duration}" if args.duration else "*"
    import subprocess

    cmd = [
        yt_dlp,
        "--no-warnings",
        "--no-playlist",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "--ffmpeg-location",
        ffmpeg,
        "-f",
        args.format,
        "-o",
        args.out,
        args.url,
    ]
    proc = subprocess.run(cmd, text=True)  # noqa: S603  argument list, no shell; yt-dlp resolved above
    if proc.returncode != 0:
        die("yt-dlp failed; try a different --format")
    print(f"downloaded {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skeleton-maker",
        description="Draw 3D body-pose skeletons on any video with the NVIDIA 3D Body Pose NIM.",
    )
    parser.add_argument("--version", action="version", version=f"skeleton-maker {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detection.add_cli(sub)
    nim.add_cli(sub)
    render.add_cli(sub)
    character.add_cli(sub)
    environment.add_cli(sub)
    envscore.add_cli(sub)

    p = sub.add_parser("clip", help="cut a window and make it NIM-conformant")
    p.add_argument("video")
    p.add_argument("--out", required=True, help="conformed clip to write")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="length in seconds")
    p.set_defaults(func=cmd_clip)

    p = sub.add_parser("download", help="fetch a window of a video with yt-dlp")
    p.add_argument("url")
    p.add_argument("--out", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--format", default="bestvideo[height<=1080][ext=mp4]+bestaudio/best[ext=mp4]")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("verify", help="check a run's artifacts")
    p.add_argument("--clip", required=True, help="the clip that was sent")
    p.add_argument("--boxes", required=True, help="the annotation that was sent")
    p.add_argument("--poses", required=True, help="the pose output from `pose`")
    p.add_argument("--overlay", default=None, help="optional overlay to compare against the clip")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("stubs", help="regenerate the vendored gRPC stubs")
    p.set_defaults(func=cmd_stubs)

    p = sub.add_parser("all", help="clip -> track -> pose -> render for one video")
    p.add_argument("video", help="a local video file")
    p.add_argument("--out", default=None, help="overlay path (default: <work>/overlay.mp4)")
    p.add_argument("--work", default=None, help="work directory (default: <video>.skeleton)")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="length in seconds")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-bodies", type=int, default=50)
    p.add_argument("--draw", choices=["2d", "3d", "both"], default="2d")
    p.add_argument("--focal-length", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument(
        "--environment",
        action="store_true",
        help="opt in to an environment scan and combined overlay",
    )
    p.add_argument(
        "--environment-backend",
        default=environment.DEFAULT_BACKEND,
        help=f"environment worker (default: {environment.DEFAULT_BACKEND})",
    )
    p.add_argument(
        "--environment-classes",
        default=None,
        help="additional environment labels, in label or label=family form",
    )
    p.add_argument(
        "--environment-geometry",
        choices=environment.GEOMETRY_MODES,
        default="auto",
        help="environment geometry mode (default: auto)",
    )
    p.add_argument(
        "--environment-sample-fps",
        type=float,
        default=2.0,
        help="environment inference frame rate (default: 2)",
    )
    p.add_argument(
        "--environment-device",
        default="auto",
        help="device used by the environment worker (default: auto)",
    )
    p.add_argument(
        "--environment-calibration",
        default=None,
        help="versioned environment camera calibration JSON",
    )
    p.add_argument(
        "--environment-cache-dir",
        default=None,
        help="environment inference cache directory",
    )
    p.add_argument(
        "--environment-no-cache",
        action="store_true",
        help="disable environment inference caching",
    )
    p.set_defaults(func=cmd_all)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
