# SPDX-License-Identifier: MIT
"""The gRPC stubs are generated from the bundled protos; prove that path works.

These tests are the reason a broken package is caught before a paid NIM call.
"""

import pytest

grpc = pytest.importorskip("grpc")


def test_bundled_protos_are_present():
    import os

    from skeleton_maker.nim import PROTO_DIR

    found = []
    for root, _dirs, files in os.walk(PROTO_DIR):
        found.extend(f for f in files if f.endswith(".proto"))
    assert "body_pose.proto" in " ".join(found), f"protos missing under {PROTO_DIR}: {found}"
    assert "service_info.proto" in " ".join(found)


def test_stubs_generate_and_import():
    pb2, pb2_grpc = __import__("skeleton_maker.nim", fromlist=["load_stubs"]).load_stubs()
    assert hasattr(pb2, "BodyPoseRequest")
    assert hasattr(pb2, "BodyPoseResponse")
    assert hasattr(pb2, "FrameBoxes")
    assert hasattr(pb2, "BoundingBox")
    assert hasattr(pb2_grpc, "BodyPoseServiceStub")


def test_request_message_accepts_config_and_boxes():
    from skeleton_maker.nim import load_stubs

    pb2, _ = load_stubs()
    req = pb2.BodyPoseRequest(
        config=pb2.BodyPoseConfig(focal_length=0.0),
        tracked_bboxes=[pb2.FrameBoxes(
            frame_id=0,
            boxes=[pb2.BoundingBox(x=1.0, y=2.0, width=3.0, height=4.0, tracking_id=7)],
        )],
    )
    assert req.config.focal_length == 0.0
    assert req.tracked_bboxes[0].boxes[0].tracking_id == 7
    assert req.video_data == b""


def test_body_to_dict_shape():
    from skeleton_maker.constants import NUM_JOINTS
    from skeleton_maker.nim import body_to_dict, load_stubs

    pb2, _ = load_stubs()
    body = pb2.Body(
        bbox=pb2.BoundingBox(x=1.0, y=2.0, width=3.0, height=4.0, tracking_id=5),
        keypoints_2d=[pb2.Point2f(x=1.0, y=2.0)] * NUM_JOINTS,
        keypoints_3d=[pb2.Point3f(x=1.0, y=2.0, z=3.0)] * NUM_JOINTS,
        keypoint_confidence=[0.5] * NUM_JOINTS,
        joint_rotations=[pb2.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)] * NUM_JOINTS,
        rest_pose=[pb2.Point3f(x=0.0, y=0.0, z=0.0)] * NUM_JOINTS,
        root_pose=pb2.Transform3f(
            translation=pb2.Point3f(x=0.0, y=0.0, z=0.0),
            rotation=pb2.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )
    out = body_to_dict(pb2, body)
    assert out["tracking_id"] == 5
    assert len(out["keypoints_2d"]) == NUM_JOINTS
    assert len(out["keypoints_2d"][0]) == 2
    assert len(out["keypoints_3d"][0]) == 3
    assert len(out["joint_rotations"][0]) == 4
    assert len(out["root_pose"]["translation"]) == 3
