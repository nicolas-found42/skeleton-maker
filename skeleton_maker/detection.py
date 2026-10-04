# SPDX-License-Identifier: MIT
"""Person detection and tracking: the boxes the NIM needs.

The 3D Body Pose NIM does not detect people. It estimates a pose once per
supplied box, so something has to say where the people are and keep a stable id
on each one across frames. Ultralytics YOLO plus ByteTrack does both.
"""

import argparse
import json
import logging
import sys

import cv2

PERSON_CLASS = 0  # COCO class index for "person"


def pick_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the fastest available backend."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception as exc:
        logging.getLogger(__name__).debug("device probe failed, using cpu: %s", exc)
    return "cpu"


def _person_boxes(result):
    """Yield (xyxy, confidence, track_id) for one frame's detections."""
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else None
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        tid = int(ids[i]) if ids is not None else -1
        yield (x1, y1, x2, y2), float(confs[i]), tid


def scan(
    video: str,
    *,
    model: str,
    imgsz: int,
    conf: float,
    interval: float,
    device: str,
    quiet: bool = False,
) -> list[dict]:
    """Sample the video every ``interval`` seconds and report person counts.

    Use this to choose a lively window before paying for NIM inference.
    """
    from ultralytics import YOLO

    net = YOLO(model)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"error: cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    step = max(1, round(interval * fps))
    for frame_idx in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        res = list(
            net.predict(
                frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS], device=device, verbose=False
            )
        )[0]
        hits = list(_person_boxes(res))
        heights = sorted(max(0.0, y2 - y1) for (_x1, y1, _x2, y2), _c, _t in hits)
        area = 0.0
        h_img, w_img = frame.shape[:2]
        for (x1, y1, x2, y2), _c, _t in hits:
            area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        rows.append(
            {
                "t": round(frame_idx / fps, 2),
                "n_people": len(hits),
                "max_box_h": round(heights[-1], 2) if heights else 0.0,
                "median_box_h": round(heights[len(heights) // 2], 2) if heights else 0.0,
                "area_frac": round(area / (h_img * w_img), 5),
            }
        )
    cap.release()
    if not quiet:
        print("t_sec,n_people,max_box_h,median_box_h,area_frac")
        for r in rows:
            print(f"{r['t']},{r['n_people']},{r['max_box_h']},{r['median_box_h']},{r['area_frac']}")
    return rows


def track(
    video: str,
    *,
    model: str,
    imgsz: int,
    conf: float,
    tracker: str,
    device: str,
    quiet: bool = False,
) -> list[dict]:
    """Track people across one clip.

    Returns a list of per-frame dicts, each ``{"frame_id", "detections": [...]}``
    with ``bbox`` as ``[x, y, w, h]`` in full-image pixels. ``frame_id`` is the
    decoded index of the clip, which is what the NIM annotation format uses.
    """
    from ultralytics import YOLO

    net = YOLO(model)
    results = net.track(
        source=video,
        stream=True,
        persist=True,
        tracker=tracker,
        imgsz=imgsz,
        conf=conf,
        classes=[PERSON_CLASS],
        device=device,
        verbose=False,
    )

    frames = []
    track_ids = set()
    for frame_id, res in enumerate(results):
        dets = []
        for (x1, y1, x2, y2), score, tid in _person_boxes(res):
            if score < conf:
                continue
            # An untracked detection is still a body: hold a slot for it so the
            # pose still runs this frame, even though its id will not persist.
            if tid < 0:
                tid = 1000 + len(dets)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            dets.append(
                {
                    "tracking_id": tid,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 4),
                }
            )
            track_ids.add(tid)
        frames.append({"frame_id": frame_id, "detections": dets})

    if not quiet:
        occupied = sum(1 for f in frames if f["detections"])
        print(
            f"tracked {len(track_ids)} ids over {len(frames)} frames "
            f"({occupied} frames have at least one person)",
            file=sys.stderr,
        )
    return frames


def add_cli(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("scan", help="sample a video and count people, to pick a window")
    p.add_argument("video", help="input video")
    p.add_argument(
        "--interval", type=float, default=2.0, help="sampling period in seconds (default: 2)"
    )
    p.add_argument("--model", default="yolo11s.pt", help="ultralytics model (default: yolo11s.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (default: 640)")
    p.add_argument("--conf", type=float, default=0.25, help="detection threshold (default: 0.25)")
    p.add_argument("--device", default="auto", help="auto|mps|0|cpu (default: auto)")

    p = subparsers.add_parser("track", help="track people and write a NIM bbox annotation")
    p.add_argument("video", help="input video")
    p.add_argument("--out-bbox", default="boxes.txt", help="annotation to write")
    p.add_argument("--out-json", default=None, help="optional tracker JSON sidecar with scores")
    p.add_argument("--model", default="yolo11s.pt", help="ultralytics model (default: yolo11s.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (default: 640)")
    p.add_argument("--conf", type=float, default=0.25, help="detection threshold (default: 0.25)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="ultralytics tracker config")
    p.add_argument("--device", default="auto", help="auto|mps|0|cpu (default: auto)")
    p.add_argument(
        "--max-bodies",
        type=int,
        default=50,
        help="keep at most this many ids, the most persistent first (default: 50)",
    )


def run_track(args) -> int:
    from .bbox import filter_frames, write_annotation

    device = pick_device(args.device)
    frames = track(
        args.video,
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        tracker=args.tracker,
        device=device,
    )
    kept = filter_frames(frames, args.max_bodies)
    summary = write_annotation(kept, args.out_bbox)
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"video": args.video, "frames": frames}, fh, indent=1)
    print(
        f"bodies={summary['bodies']} rows={summary['rows']} "
        f"coverage={summary['coverage']:.1f}% "
        f"frames_with_boxes={summary['frames_covered']}/{summary['frames_total']} "
        f"max_boxes_in_a_frame={summary['max_per_frame']}"
    )
    if summary["empty_frames"]:
        print(f"note: no box in frames {summary['empty_frames']} -- these frames will have no pose")
    return 0


def run_scan(args) -> int:
    scan(
        args.video,
        model=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        interval=args.interval,
        device=pick_device(args.device),
    )
    return 0
