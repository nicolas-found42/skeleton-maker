# SPDX-License-Identifier: MIT
"""The TUM RGB-D converter, checked on a synthetic sequence whose geometry is known exactly.

The camera looks straight at a plane 2 m away and slides sideways 1 cm per frame, so every
expected world position and distance below is worked out from the pinhole model, never from
the converter's own formulas.
"""

import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from skeleton_maker.envannotations import load_annotation
from skeleton_maker.envgeometry import load_reference

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_tum_reference.py"
_spec = importlib.util.spec_from_file_location("build_tum_reference", SCRIPT)
assert _spec is not None
assert _spec.loader is not None
tum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tum)

W, H, FX, CX, CY = 96, 72, 100.0, 48.0, 36.0
CAMERA = {"size": [W, H], "intrinsics": [FX, FX, CX, CY], "distortion": [0.0] * 5}
PLANE_M = 2.0
STEP_M = 0.01
FRAMES = 120


def _sequence(root: Path, *, poses=None, depth_ok=True, late_depth=()):
    """A sliding camera in front of a blocky plane; ``poses`` overrides the trajectory."""
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rng = np.random.default_rng(7)
    blocks = rng.integers(0, 2, size=(H // 6, (W + FRAMES) // 6 + 2)) * 200 + 20
    texture = np.kron(blocks, np.ones((6, 6))).astype(np.uint8)
    rgb, depth, truth = [], [], []
    for i in range(FRAMES):
        t = 1000.0 + i / 30.0
        shift = i  # one pixel per frame: 1 cm * 100 px / 2 m = 0.5 px; any shift keeps corners
        frame = texture[:H, shift : shift + W]
        cv2.imwrite(str(root / "rgb" / f"{t:.6f}.png"), cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        rgb.append(f"{t:.6f} rgb/{t:.6f}.png")
        td = t + (1.0 if i in late_depth else 0.005)
        z = np.full((H, W), PLANE_M * 5000 if depth_ok else 0, np.uint16)
        cv2.imwrite(str(root / "depth" / f"{td:.6f}.png"), z)
        depth.append(f"{td:.6f} depth/{td:.6f}.png")
    for k in range(FRAMES * 4):
        t = 1000.0 + k / 120.0
        i = k / 4.0
        x, qy, qw = (STEP_M * i, 0.0, 1.0) if poses is None else poses(i)
        truth.append(f"{t:.6f} {x:.6f} 0 0 0 {qy:.6f} 0 {qw:.6f}")
    for name, rows in (("rgb", rgb), ("depth", depth), ("groundtruth", truth)):
        (root / f"{name}.txt").write_text("# header\n" + "\n".join(rows) + "\n")
    return root


def _build(tmp_path, **kwargs):
    root = _sequence(tmp_path / "seq", **kwargs.pop("sequence", {}))
    return tum.build(
        root,
        tmp_path / "out",
        name="synthetic",
        split="heldout",
        sample_step=10,
        camera=CAMERA,
        encode=False,
        clip_sha256="a" * 64,
        **kwargs,
    )


def test_quaternion_matrix_turns_x_into_y_for_a_quarter_turn_about_z():
    s = math.sqrt(0.5)
    matrix = tum.quaternion_matrix(0.0, 0.0, s, s)

    assert matrix @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_anchors_are_measured_positions_on_the_plane(tmp_path):
    result = _build(tmp_path)

    for anchor in result["reference"]["anchors"]:
        u, v = anchor["pixel"]
        frame_x = STEP_M * anchor["frame_id"]
        expected = [frame_x + (u - CX) / FX * PLANE_M, (v - CY) / FX * PLANE_M, PLANE_M]
        assert anchor["position_m"] == pytest.approx(expected, abs=2e-3)  # pose sampled at 120 Hz


def test_the_check_dimension_is_the_distance_between_two_measured_points(tmp_path):
    result = _build(tmp_path)

    dim = result["reference"]["measured_dimensions"][0]
    (ua, va), (ub, vb) = dim["a"]["pixel"], dim["b"]["pixel"]
    assert dim["a"]["frame_id"] == dim["b"]["frame_id"]
    assert dim["length_m"] == pytest.approx(math.hypot(ub - ua, vb - va) / FX * PLANE_M, abs=1e-6)
    assert dim["length_m"] >= tum.MIN_DIMENSION_M
    assert dim["role"] == "check"


def test_the_outputs_satisfy_the_projects_own_loaders(tmp_path):
    result = _build(tmp_path)
    out = tmp_path / "out"

    reference = load_reference(
        out / "synthetic.geometry-reference.json", width=W, height=H, frame_count=FRAMES
    )
    annotation = load_annotation(out / "synthetic.annotation.json")
    assert reference is not None

    roles = [a["role"] for a in reference["anchors"]]
    assert roles.count("fit") >= 3
    assert roles.count("check") >= 1
    fit = {(a["frame_id"], tuple(a["pixel"])) for a in reference["anchors"] if a["role"] == "fit"}
    check = {
        (a["frame_id"], tuple(a["pixel"])) for a in reference["anchors"] if a["role"] == "check"
    }
    assert fit.isdisjoint(check)
    assert annotation["scope"] == "geometry"
    assert annotation["frames"] == []
    assert annotation["geometry"]["dimensions"][0]["withheld"] is True
    assert {cp["frame_id"] for cp in annotation["geometry"]["control_points"]} == set(
        annotation["geometry"]["eligible_frames"]
    )
    assert result["annotation"]["provenance"]["kind"] == "published_dataset"


def test_the_clip_can_be_cut_to_a_bounded_number_of_samples(tmp_path):
    result = _build(tmp_path, max_samples=8)

    assert result["report"]["frames"] == 71  # (8 - 1) * 10 + 1
    assert result["annotation"]["shots"] == [{"first_frame": 0, "last_frame": 70}]
    assert all(a["frame_id"] <= 70 for a in result["reference"]["anchors"])
    assert result["report"]["parameters"]["max_samples"] == 8


def test_anchor_frames_lie_on_the_sampling_grid(tmp_path):
    result = _build(tmp_path)

    assert all(a["frame_id"] % 10 == 0 for a in result["reference"]["anchors"])


def test_depth_and_pose_agree_exactly_on_a_flat_plane(tmp_path):
    result = _build(tmp_path)

    agreement = result["report"]["depth_pose_agreement"]
    assert agreement["points"] > 100
    assert agreement["median_abs_error_m"] == pytest.approx(0.0, abs=1e-6)


def test_a_sliding_camera_is_tagged_translating(tmp_path):
    result = _build(tmp_path)

    assert "translating" in result["annotation"]["tags"]
    assert "underconstrained" not in result["annotation"]["tags"]


def test_a_static_camera_is_tagged_tripod_and_underconstrained(tmp_path):
    result = _build(tmp_path, sequence={"poses": lambda _i: (0.0, 0.0, 1.0)})

    assert {"tripod", "underconstrained"} <= set(result["annotation"]["tags"])


def test_a_turning_camera_is_tagged_pan_and_underconstrained(tmp_path):
    def turning(i):
        half = math.radians(40.0 * i / FRAMES) / 2  # up to 40 degrees about the y axis
        return 0.0, math.sin(half), math.cos(half)

    result = _build(tmp_path, sequence={"poses": turning})

    assert {"pan", "underconstrained"} <= set(result["annotation"]["tags"])
    assert "translating" not in result["annotation"]["tags"]
    assert "motion_blur" not in result["annotation"]["tags"]  # 10 degrees a second is slow


def test_a_fast_turning_camera_is_tagged_motion_blur_from_its_measured_turn_rate():
    times = np.arange(0.0, 4.0, 0.01)  # a 100 Hz motion-capture trace turning 150 degrees a second
    half = np.radians(150.0 * times) / 2
    zero = np.zeros_like(times)
    # columns: time, tx ty tz, qx qy qz qw (a turn about the y axis)
    poses = np.stack([times, zero, zero, zero, zero, np.sin(half), zero, np.cos(half)], axis=1)

    motion = tum.camera_motion(poses)

    assert motion["peak_turn_rate_deg_s"] == pytest.approx(150.0, rel=0.05)
    assert motion["fast_turn_fraction"] > 0.9
    assert "motion_blur" in tum.motion_tags(motion)


def _full_size_sequence(root: Path) -> Path:
    """A real-size 640x480 synthetic sequence: a slow sideways slide in front of a textured plane."""
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rng = np.random.default_rng(11)
    texture = np.kron(rng.integers(0, 2, size=(60, 110)) * 200 + 20, np.ones((8, 8))).astype(
        np.uint8
    )
    rgb, depth, truth = [], [], []
    for i in range(240):
        t = 1000.0 + i / 30.0
        frame = texture[:480, i : i + 640]
        cv2.imwrite(str(root / "rgb" / f"{t:.6f}.png"), cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        rgb.append(f"{t:.6f} rgb/{t:.6f}.png")
        td = t + 0.005
        cv2.imwrite(str(root / "depth" / f"{td:.6f}.png"), np.full((480, 640), 10000, np.uint16))
        depth.append(f"{td:.6f} depth/{td:.6f}.png")
        truth.append(f"{t:.6f} {0.01 * i:.6f} 0 0 0 0 0 1")
    for name, rows in (("rgb", rgb), ("depth", depth), ("groundtruth", truth)):
        (root / f"{name}.txt").write_text("# header\n" + "\n".join(rows) + "\n")
    return root


def test_the_cli_default_camera_is_freiburg1_so_existing_calls_are_unchanged(tmp_path):
    """Without --camera the CLI behaves as before and says so in its report."""
    root, out = _full_size_sequence(tmp_path / "seq"), tmp_path / "out"

    exit_normal = tum.main(
        [str(root), str(out), "--name", "default-cam", "--split", "heldout", "--sample-step", "10"]
    )

    report = json.loads((out / "default-cam.build-report.json").read_text())
    reference = json.loads((out / "default-cam.geometry-reference.json").read_text())
    annotation = json.loads((out / "default-cam.annotation.json").read_text())

    assert exit_normal == 0
    assert report["parameters"]["camera"] == "freiburg1"
    assert reference["camera"]["intrinsics"][0][0] == 517.3
    assert annotation["frames"] == []


def test_the_cli_selects_the_freiburg2_camera(tmp_path):
    """A real-size fr2 sequence: 640x480 frames, Freiburg 2 intrinsics, a slow slide."""
    root, out = _full_size_sequence(tmp_path / "seq"), tmp_path / "out"

    exit_normal = tum.main(
        [str(root), str(out), "--name", "pan", "--split", "development", "--camera", "freiburg2"]
    )
    report = json.loads((out / "pan.build-report.json").read_text())
    reference = json.loads((out / "pan.geometry-reference.json").read_text())
    annotation = json.loads((out / "pan.annotation.json").read_text())

    assert exit_normal == 0
    assert report["parameters"]["camera"] == "freiburg2"
    assert reference["camera"]["intrinsics"][0][0] == 520.9
    assert annotation["frames"] == []


def test_the_cli_refuses_an_unknown_camera(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        tum.main(
            [
                "/nonexistent",
                str(tmp_path / "out"),
                "--name",
                "x",
                "--split",
                "development",
                "--camera",
                "kinect-xyz",
            ]
        )

    assert excinfo.value.code != 0


def test_frames_without_a_close_depth_image_are_never_anchors(tmp_path):
    late = range(0, 30)
    result = _build(tmp_path, sequence={"late_depth": late})

    assert result["report"]["frames_without_association"] == 30
    assert all(a["frame_id"] >= 30 for a in result["reference"]["anchors"])


def test_a_sequence_whose_depth_and_pose_disagree_is_refused(tmp_path):
    def yawing(i):
        half = math.radians(1.0 * i) / 2  # a degree a frame, while the depth never changes
        return 0.0, math.sin(half), math.cos(half)

    with pytest.raises(tum.ReferenceError_, match="disagree"):
        _build(tmp_path, sequence={"poses": yawing})


def test_a_sequence_without_depth_cannot_yield_anchors(tmp_path):
    with pytest.raises(tum.ReferenceError_, match="usable corners"):
        _build(tmp_path, sequence={"depth_ok": False})
