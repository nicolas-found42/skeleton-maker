# SPDX-License-Identifier: MIT
"""Constants, including the Nova-77 skeleton topology.

NOVA77_PARENTS, TRACK_COLORS, COLOR_2D and COLOR_3D are reproduced from the
NVIDIA-Maxine/nim-clients body-pose sample client (MIT); see NOTICE.
"""

# --- NIM limits and protocol constants -------------------------------------

#: The annotation header must be 1..50 tracked bodies, and no frame may carry
#: more than this many boxes. The NIM rejects the request otherwise.
MAX_BODIES = 50

#: The exact quadruple that marks a tracked body as absent in a frame.
ABSENT_BBOX = (-1.0, -1.0, -1.0, -1.0)

#: Compressed video is streamed to the NIM in chunks of this size.
DATA_CHUNK_SIZE = 256 * 1024

#: Default RPC deadline, generous enough for long clips and many bodies.
DEFAULT_TIMEOUT = 3600.0

#: Hosted 3D Body Pose function on NVIDIA Cloud Functions.
DEFAULT_TARGET = "grpc.nvcf.nvidia.com:443"
DEFAULT_FUNCTION_ID = "254b3621-bb66-4b7b-9c89-621c352b4d91"

#: Per-joint array length in the Nova-77 skeleton order.
NUM_JOINTS = 77

#: Container extensions the NIM accepts.
VIDEO_FILE_TYPES = ["mp4", "webm", "mkv"]

# --- Video encoding the NIM requires ---------------------------------------

#: The NIM enforces a constant frame rate and 4:2:0 8-bit SDR video. These are
#: the ffmpeg arguments that produce a conforming file.
CFR_FPS = 30
CONFORM_ENCODE_ARGS = [
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
    "-r",
    str(CFR_FPS),
    "-fps_mode",
    "cfr",
    "-an",
    "-movflags",
    "+faststart",
]

# --- Nova-77 skeleton, mirroring the AR SDK sample BodyPose3DApp -----------

#: Parent joint of each of the 77 joints; index 0 (Hips) is the root.
NOVA77_PARENTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    3,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    14,
    19,
    20,
    21,
    22,
    14,
    24,
    25,
    26,
    27,
    14,
    29,
    30,
    31,
    32,
    14,
    34,
    35,
    36,
    37,
    3,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    42,
    47,
    48,
    49,
    50,
    42,
    52,
    53,
    54,
    55,
    42,
    57,
    58,
    59,
    60,
    42,
    62,
    63,
    64,
    65,
    0,
    67,
    68,
    69,
    70,
    0,
    72,
    73,
    74,
    75,
]

#: (child, parent) bone list, root excluded.
NOVA77_SKELETON_LINKS = [(i, p) for i, p in enumerate(NOVA77_PARENTS) if p >= 0]

#: Per-tracking-id BGR colours for bounding boxes and ID labels.
TRACK_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]

COLOR_2D = (0, 255, 0)  # 2D keypoints and skeleton, green (BGR)
COLOR_3D = (0, 0, 255)  # reprojected 3D keypoints and skeleton, red (BGR)

#: Keypoints landing outside their box by this fraction of its size still draw.
BBOX_MARGIN = 0.25

#: Which keypoint sets the overlay draws.
DRAW_KEYPOINTS_CONFIGS = {
    "2d": (True, False),
    "3d": (False, True),
    "both": (True, True),
}
