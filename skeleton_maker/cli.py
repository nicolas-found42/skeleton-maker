# SPDX-License-Identifier: MIT
"""``skeleton-maker`` command line: scan -> clip -> track -> pose -> render.

Each stage is also usable on its own; ``all`` chains them for a local video.
"""

import argparse
import os
import shutil
import sys

from . import __version__, bbox, detection, nim, render, verify
from .constants import CFR_FPS, CONFORM_ENCODE_ARGS
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
    ff_args += CONFORM_ENCODE_ARGS + [args.out, "-y"]
    run_ffmpeg(ff_args, f"cutting {args.out}")

    out = probe_video(args.out)
    print(f"source : {src['width']}x{src['height']} {src['codec']} "
          f"{src['r_fps']:.3f}fps {src['duration']:.2f}s")
    print(f"clip   : {out['width']}x{out['height']} {out['codec']} "
          f"{out['pix_fmt']} {out['r_fps']:.3f}fps {out['duration']:.2f}s "
          f"{out['size'] / 1e6:.1f}MB -> {args.out}")

    notes = []
    if out["size"] > 50_000_000:
        notes.append(f"the hosted endpoint caps input at 50MB; this clip is "
                     f"{out['size'] / 1e6:.1f}MB. Use a shorter window, lower "
                     f"resolution, or higher --crf.")
    if abs(out["r_fps"] - out["avg_fps"]) > 0.01:
        notes.append(f"frame rate is not constant ({out['r_fps']:.3f} vs "
                     f"{out['avg_fps']:.3f}); the NIM rejects variable frame rate")
    if not is_streamable_mp4(args.out):
        notes.append("the MP4 is not streamable (moov after mdat), so the NIM will "
                     "buffer the whole file before processing")
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
    work = args.work or (os.path.splitext(args.video)[0] + ".skeleton")
    os.makedirs(work, exist_ok=True)
    clip = os.path.join(work, "clip.mp4")
    boxes = os.path.join(work, "boxes.txt")
    poses = os.path.join(work, "pose.json")
    overlay = args.out or os.path.join(work, "overlay.mp4")

    print(f"== work directory: {work}")
    print("== 1/4 conforming the clip")
    rc = cmd_clip(argparse.Namespace(video=args.video, out=clip,
                                     start=args.start, duration=args.duration))
    if rc:
        return rc

    print("== 2/4 detecting and tracking people")
    rc = detection.run_track(argparse.Namespace(
        video=clip, out_bbox=boxes, out_json=None,
        model=args.model, imgsz=args.imgsz, conf=args.conf,
        tracker=args.tracker, device=args.device, max_bodies=args.max_bodies))
    if rc:
        return rc

    print("== 3/4 estimating poses with the NIM")
    summary = nim.run(clip, boxes, poses, focal_length=args.focal_length,
                      timeout=args.timeout)
    print(f"   {summary['frames']} frames ({summary['frames_with_bodies']} with bodies) "
          f"in {summary['seconds']}s")

    print("== 4/4 rendering the overlay")
    render.render(clip, poses, overlay, draw=args.draw,
                  focal_length=args.focal_length or summary["focal_length"])

    print("== verifying")
    ok = verify.run(clip, boxes, poses, overlay, report=True)
    print(f"\ndone. work directory: {work}")
    return 0 if ok else 1


def cmd_download(args) -> int:
    """Grab a window of a video with yt-dlp, NIM-conformant."""
    require_tool("yt-dlp", "Install it: uv pip install yt-dlp")
    ffmpeg = require_tool("ffmpeg")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    section = f"*{args.start}-{args.start + args.duration}" if args.duration else "*"
    import subprocess

    cmd = [shutil.which("yt-dlp"), "--no-warnings", "--no-playlist",
           "--download-sections", section, "--force-keyframes-at-cuts",
           "--ffmpeg-location", ffmpeg,
           "-f", args.format, "-o", args.out, args.url]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        die("yt-dlp failed; try a different --format")
    print(f"downloaded {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skeleton-maker",
        description="Draw 3D body-pose skeletons on any video with the NVIDIA "
                    "3D Body Pose NIM.",
    )
    parser.add_argument("--version", action="version", version=f"skeleton-maker {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detection.add_cli(sub)
    nim.add_cli(sub)
    render.add_cli(sub)

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
    p.set_defaults(func=cmd_all)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
