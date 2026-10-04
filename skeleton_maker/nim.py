# SPDX-License-Identifier: MIT
"""gRPC client for the NVIDIA 3D Body Pose NIM.

The NIM exposes one bidirectional stream. The client sends a first message with
the configuration and the *whole* tracked-box annotation, then streams the
compressed video in chunks. The server decodes the video on its own GPU and
streams back frame-aligned poses, delayed by its ~120-frame temporal window.
"""

import importlib
import json
import os
import sys
import time

from .bbox import read_annotation
from .constants import (
    DATA_CHUNK_SIZE,
    DEFAULT_FUNCTION_ID,
    DEFAULT_TARGET,
    DEFAULT_TIMEOUT,
    NUM_JOINTS,
)

GEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gen")
PROTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto")


def _generate_stubs() -> None:
    """Compile the bundled protos into ``_gen`` on first use."""
    try:
        from grpc_tools import protoc
    except ImportError:
        raise SystemExit(
            "error: the gRPC stubs are not present and grpcio-tools is missing.\n"
            "       Install it:  pip install grpcio-tools\n"
            "       Or regenerate manually:  python -m skeleton_maker.grpc_stubs"
        ) from None
    import grpc_tools

    well_known = os.path.join(os.path.dirname(grpc_tools.__file__), "_proto")
    os.makedirs(GEN_DIR, exist_ok=True)
    protos = []
    for root, _dirs, files in os.walk(PROTO_DIR):
        protos.extend(os.path.join(root, f) for f in files if f.endswith(".proto"))
    code = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"-I{well_known}",
            f"--python_out={GEN_DIR}",
            f"--grpc_python_out={GEN_DIR}",
            *protos,
        ]
    )
    if code != 0:
        raise SystemExit(f"error: protoc failed with code {code}")


def load_stubs():
    """Import the generated stubs, generating them if they are missing."""
    if GEN_DIR not in sys.path:
        sys.path.insert(0, GEN_DIR)
    try:
        pb2 = importlib.import_module("nvidia.ai4m.body_pose.v1.body_pose_pb2")
        pb2_grpc = importlib.import_module("nvidia.ai4m.body_pose.v1.body_pose_pb2_grpc")
    except ImportError:
        _generate_stubs()
        importlib.invalidate_caches()
        pb2 = importlib.import_module("nvidia.ai4m.body_pose.v1.body_pose_pb2")
        pb2_grpc = importlib.import_module("nvidia.ai4m.body_pose.v1.body_pose_pb2_grpc")
    return pb2, pb2_grpc


def body_to_dict(pb2, body) -> dict:
    """Convert one posed body to a plain, JSON-serialisable dict."""
    root = body.root_pose
    return {
        "tracking_id": body.bbox.tracking_id,
        "bbox": [body.bbox.x, body.bbox.y, body.bbox.width, body.bbox.height],
        "keypoints_2d": [[p.x, p.y] for p in body.keypoints_2d],
        "keypoints_confidence": list(body.keypoint_confidence),
        "keypoints_3d": [[p.x, p.y, p.z] for p in body.keypoints_3d],
        "rest_pose": [[p.x, p.y, p.z] for p in body.rest_pose],
        "joint_rotations": [[q.x, q.y, q.z, q.w] for q in body.joint_rotations],
        "root_pose": {
            "translation": [root.translation.x, root.translation.y, root.translation.z],
            "rotation": [root.rotation.x, root.rotation.y, root.rotation.z, root.rotation.w],
        },
    }


def _read_chunks(path: str, chunk_size: int = DATA_CHUNK_SIZE):
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                return
            yield chunk


def run(
    video: str,
    bbox: str,
    output: str,
    *,
    api_key: str | None = None,
    target: str = DEFAULT_TARGET,
    function_id: str = DEFAULT_FUNCTION_ID,
    focal_length: float = 0.0,
    enable_contact: bool | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    session_id: str | None = None,
    attempts: int = 3,
    verbose: bool = True,
) -> dict:
    """Send one clip to the NIM and write the poses as JSON Lines.

    Returns a summary dict. ``output`` gets one JSON object per frame, including
    frames with no bodies, so the file is directly frame-indexable.
    """
    import grpc

    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(
            "error: no API key. Set NVIDIA_API_KEY, or pass --api-key.\n"
            "       Get one at https://build.nvidia.com/nvidia/body-pose"
        )

    pb2, pb2_grpc = load_stubs()
    boxes_by_frame = read_annotation(bbox)

    def requests():
        pose_config = pb2.BodyPoseConfig(focal_length=focal_length)
        if enable_contact is not None:
            pose_config.enable_contact = enable_contact
        frame_boxes = [
            pb2.FrameBoxes(
                frame_id=fid,
                boxes=[
                    pb2.BoundingBox(x=x, y=y, width=w, height=h, tracking_id=tid)
                    for tid, x, y, w, h in rows
                ],
            )
            for fid, rows in sorted(boxes_by_frame.items())
        ]
        yield pb2.BodyPoseRequest(config=pose_config, tracked_bboxes=frame_boxes)
        for chunk in _read_chunks(video):
            yield pb2.BodyPoseRequest(video_data=chunk)

    metadata = (("authorization", f"Bearer {api_key}"), ("function-id", function_id))
    if session_id:
        metadata += (("client-session-id", session_id),)

    # The hosted endpoint can fail to place a worker (UNAVAILABLE, or a
    # DEADLINE_EXCEEDED naming "failed to establish link to worker"). That is
    # transient and worth another attempt: the request is a read-only inference.
    transient = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
    started = time.time()
    for attempt in range(1, attempts + 1):
        channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
        stub = pb2_grpc.BodyPoseServiceStub(channel)
        frames_written = 0
        frames_with_bodies = 0
        focal_seen = 0.0
        service_info = None
        out = None
        try:
            responses = stub.EstimateBodyPose(
                requests(), metadata=metadata, timeout=timeout or None
            )
            for key, value in responses.initial_metadata() or ():
                if key == "warning" and verbose:
                    print(f"server warning: {value}", file=sys.stderr)
            for response in responses:
                if response.HasField("service_info"):
                    service_info = {
                        "feature_name": response.service_info.feature_name,
                        "feature_version": response.service_info.feature_version,
                        "model_info": response.service_info.model_info,
                        "server_request_id": response.service_info.server_request_id,
                    }
                    if verbose:
                        print(
                            f"NIM {service_info['feature_name']} "
                            f"{service_info['feature_version']} "
                            f"({service_info['model_info']}), "
                            f"request {service_info['server_request_id']}",
                            file=sys.stderr,
                        )
                    continue
                if response.focal_length > 0.0:
                    focal_seen = response.focal_length
                if not response.ready:
                    continue
                if out is None:
                    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
                    out = open(output, "w")  # noqa: SIM115  opened on the first response, closed in the finally block
                detections = [body_to_dict(pb2, b) for b in response.bodies]
                out.write(
                    json.dumps({"frame_id": response.frame_id, "detections": detections}) + "\n"
                )
                frames_written += 1
                if detections:
                    frames_with_bodies += 1
                if verbose and frames_written % 120 == 0:
                    print(f"  {frames_written} frames ...", file=sys.stderr)
                if response.stream_flushed:
                    break
        except grpc.RpcError as err:
            code = err.code()
            detail = (err.details() or "").strip()
            if code in transient and attempt < attempts:
                if verbose:
                    print(
                        f"attempt {attempt}/{attempts} failed ({code.name}: {detail}); retrying",
                        file=sys.stderr,
                    )
                time.sleep(min(2**attempt, 10))
                continue
            if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                raise SystemExit(
                    f"error: the NIM is at capacity ({detail}). Wait and retry, or "
                    "lower --max-bodies to shorten the stream."
                ) from None
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise SystemExit(
                    f"error: the NIM rejected the input -- {detail}\n"
                    "       Check the clip is H.264 yuv420p at a constant frame rate, "
                    "under 50MB, and that the annotation came from this same clip."
                ) from None
            raise SystemExit(f"error: NIM call failed ({code.name}): {detail}") from None
        finally:
            if out is not None:
                out.close()
            channel.close()

        if frames_written == 0:
            if attempt < attempts:
                if verbose:
                    print(
                        f"attempt {attempt}/{attempts} returned no poses; retrying", file=sys.stderr
                    )
                continue
            raise SystemExit(
                f"error: the NIM returned no poses after {attempts} attempts; check the server log"
            )
        break

    return {
        "frames": frames_written,
        "frames_with_bodies": frames_with_bodies,
        "focal_length": focal_seen,
        "service_info": service_info,
        "seconds": round(time.time() - started, 2),
        "attempts": attempt,
    }


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("pose", help="run the NIM on a clip and write poses")
    p.add_argument("video", help="input video (MP4/H.264, constant frame rate)")
    p.add_argument("bbox", help="tracked bounding-box annotation from `track`")
    p.add_argument("--out", required=True, help="pose output path (JSON Lines)")
    p.add_argument(
        "--target", default=DEFAULT_TARGET, help=f"gRPC endpoint (default: {DEFAULT_TARGET})"
    )
    p.add_argument("--function-id", default=DEFAULT_FUNCTION_ID, help="NVCF function id")
    p.add_argument("--api-key", default=None, help="NVIDIA API key (default: $NVIDIA_API_KEY)")
    p.add_argument(
        "--focal-length",
        type=float,
        default=0.0,
        help="pinhole focal length in pixels; 0 asks for the server default",
    )
    contact = p.add_mutually_exclusive_group()
    contact.add_argument(
        "--enable-contact",
        dest="enable_contact",
        action="store_true",
        default=None,
        help="request static-camera contact correction (slower)",
    )
    contact.add_argument(
        "--no-enable-contact",
        dest="enable_contact",
        action="store_false",
        help="request no contact correction",
    )
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="RPC deadline in seconds")
    p.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="retries when the hosted endpoint cannot place a worker (default: 3)",
    )
    p.add_argument("--session-id", default=None, help="your own session id, echoed by the server")


def run_cli(args) -> int:
    summary = run(
        args.video,
        args.bbox,
        args.out,
        api_key=args.api_key,
        target=args.target,
        function_id=args.function_id,
        focal_length=args.focal_length,
        enable_contact=args.enable_contact,
        timeout=args.timeout,
        session_id=args.session_id,
        attempts=args.attempts,
    )
    print(
        f"wrote {args.out}: {summary['frames']} frames "
        f"({summary['frames_with_bodies']} with bodies) in {summary['seconds']}s"
    )
    return 0


__all__ = ["NUM_JOINTS", "add_cli", "body_to_dict", "load_stubs", "run", "run_cli"]
