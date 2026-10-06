# SPDX-License-Identifier: MIT
"""Lightweight shape and pixel-transform contracts for the optional DA3 worker."""

import json
import sys
import types

import numpy as np
import pytest

from skeleton_maker import envregistration
from skeleton_maker.workers import da3_geometry


def test_depth_batch_accepts_finite_native_processed_resolution():
    depth = np.ones((5, 280, 504), dtype=np.float32)

    result = da3_geometry._depth_batch(depth, expected_frames=5)

    assert result.shape == (5, 280, 504)


def test_depth_batch_rejects_wrong_frame_count_or_nonfinite_values():
    with pytest.raises(RuntimeError, match="4 depths for 5 images"):
        da3_geometry._depth_batch(np.ones((4, 280, 504)), expected_frames=5)
    with pytest.raises(RuntimeError, match="non-finite"):
        da3_geometry._depth_batch(np.full((5, 280, 504), np.nan), expected_frames=5)


def test_intrinsics_and_pixel_maps_use_actual_x_and_y_resize_ratios():
    source_width, source_height = 640, 360
    processed_width, processed_height = 504, 280
    scale_x = processed_width / source_width
    scale_y = processed_height / source_height
    processed_intrinsics = np.asarray([[315.0, 0.0, 252.0], [0.0, 175.0, 140.0], [0.0, 0.0, 1.0]])

    source_intrinsics = da3_geometry._intrinsics_in_source_space(
        processed_intrinsics, scale_x=scale_x, scale_y=scale_y
    )
    source_to_processed, processed_to_source = da3_geometry._resize_pixel_maps(
        scale_x=scale_x, scale_y=scale_y
    )

    assert source_intrinsics[0, 0] == pytest.approx(400.0)
    assert source_intrinsics[1, 1] == pytest.approx(225.0)
    assert source_intrinsics[0, 2] == pytest.approx(320.0)
    assert source_intrinsics[1, 2] == pytest.approx(180.0)
    assert source_to_processed[0][0] == pytest.approx(504 / 640)
    assert source_to_processed[1][1] == pytest.approx(280 / 360)
    assert source_to_processed[0][0] != source_to_processed[1][1]
    assert processed_to_source[0][0] == pytest.approx(640 / 504)
    assert processed_to_source[1][1] == pytest.approx(360 / 280)


def test_dynamic_source_mask_maps_to_native_depth_grid():
    source_mask = np.zeros((360, 640), dtype=np.uint8)
    source_mask[160:200, 300:340] = 255

    processed = da3_geometry._mask_to_processed(
        source_mask,
        width=640,
        height=360,
        scale_x=504 / 640,
        scale_y=280 / 360,
        lens_transform={"applied": False},
    )

    assert processed.shape == (280, 504)
    assert processed[140, 252]
    assert processed.sum() < source_mask.sum()


def test_dynamic_mask_uses_opencv_legacy_nearest_rounding_at_resize_boundary():
    source_mask = np.zeros((360, 640), dtype=np.uint8)
    source_mask[26, :] = 255

    processed = da3_geometry._mask_to_processed(
        source_mask,
        width=640,
        height=360,
        scale_x=504 / 640,
        scale_y=280 / 360,
        lens_transform={"applied": False},
    )

    # At output y=21, exact inverse scaling gives source y=27, but OpenCV's
    # legacy INTER_NEAREST resampler selects source row 26 at this boundary.
    assert 21 * 360 / 280 == 27
    assert processed[21].all()
    assert not processed[22].any()


def test_dynamic_masked_depth_pixels_are_invalidated():
    depth = np.ones((5, 8), dtype=np.float32)
    exclusion = np.zeros((5, 8), dtype=bool)
    exclusion[1:3, 4:7] = True

    filtered = da3_geometry._exclude_masked_depth(depth, exclusion)

    assert np.isnan(filtered[1:3, 4:7]).all()
    assert np.isfinite(filtered[~exclusion]).all()
    assert np.isfinite(depth).all()


def test_public_worker_keeps_skewed_distortion_image_mask_and_control_pixels_aligned(
    tmp_path, monkeypatch
):
    import cv2

    intrinsics = np.array([[800.0, 120.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    coefficients = np.array([0.1, -0.05, 0.001, 0.002, 0.0])
    ideal_xyz = np.array([[0.2, 0.1, 1.0]])
    distorted, _ = cv2.projectPoints(ideal_xyz, np.zeros(3), np.zeros(3), np.eye(3), coefficients)
    source_h = intrinsics @ np.r_[distorted.reshape(2), 1.0]
    source_pixel = np.rint(source_h[:2] / source_h[2]).astype(int)
    source_image = np.zeros((480, 640, 3), dtype=np.uint8)
    source_image[source_pixel[1], source_pixel[0], 0] = 255
    observed = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path):
            return cls()

        def to(self, *, device):
            assert device == "cpu"
            return self

        def inference(self, images, *, intrinsics, **_kwargs):
            observed["image"] = np.asarray(images[0])
            observed["intrinsics"] = np.asarray(intrinsics[0])
            return types.SimpleNamespace(
                depth=np.ones((1, 480, 640), dtype=np.float32),
                intrinsics=np.asarray(intrinsics),
                extrinsics=np.eye(4, dtype=np.float32)[None, :3],
            )

    package = types.ModuleType("depth_anything_3")
    package.__path__ = []
    api = types.ModuleType("depth_anything_3.api")
    api.__dict__["DepthAnything3"] = FakeModel
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    monkeypatch.setattr(da3_geometry, "_install_inference_only_adapters", lambda: None)
    monkeypatch.setattr(da3_geometry, "_read_frames", lambda *_: iter([(0, source_image)]))
    monkeypatch.setattr(da3_geometry, "model_dir", lambda: tmp_path)
    monkeypatch.setattr(da3_geometry, "identity", lambda: {"model": "fixture"})
    monkeypatch.setattr(da3_geometry, "_peak_rss_bytes", lambda: 0)
    monkeypatch.setattr(da3_geometry, "_mps_memory_snapshot", lambda _device: None)

    request = {
        "video": "synthetic.mp4",
        "frames": [{"frame_id": 0}],
        "shots": [{"id": "shot-0", "first_frame": 0, "last_frame": 0}],
        "device": "cpu",
        "geometry_reference": {
            "camera": {
                "intrinsics": intrinsics.tolist(),
                "distortion": {
                    "model": "opencv-brown-conrady",
                    "coefficients": coefficients.tolist(),
                },
            }
        },
        "geometry_filter": {
            "excluded_pixel_regions": [
                {
                    "frame_id": 0,
                    "bbox": [
                        float(source_pixel[0]),
                        float(source_pixel[1]),
                        float(source_pixel[0] + 1),
                        float(source_pixel[1] + 1),
                    ],
                }
            ]
        },
    }
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    assets = tmp_path / "assets"
    assets.mkdir()
    request_path.write_text(json.dumps(request))

    assert da3_geometry.run(str(request_path), str(assets), str(response_path)) == 0
    response = json.loads(response_path.read_text())
    frame = response["frames"][0]
    rgb_peak = np.array(np.unravel_index(observed["image"][:, :, 0].argmax(), (480, 640)))[::-1]
    depth = np.load(assets / frame["depth_asset"], allow_pickle=False)
    masked = np.array(np.unravel_index(np.isnan(depth).argmax(), depth.shape))[::-1]
    lens_frame = {"preprocessing": {"lens_transform": frame["preprocessing"]["lens_transform"]}}
    control_xy = envregistration._source_pixel_to_depth(
        lens_frame,
        source_pixel.astype(float).tolist(),
        request["geometry_reference"]["camera"],
        np.eye(3),
    )

    assert np.array_equal(rgb_peak, masked)
    assert rgb_peak == pytest.approx(control_xy, abs=1.0)
    output_h = frame["preprocessing"]["lens_transform"]["output_intrinsics"]
    expected = np.asarray(output_h) @ np.array([0.2, 0.1, 1.0])
    assert rgb_peak == pytest.approx(expected[:2], abs=1.0)


def test_zero_skew_brown_conrady_image_remap_matches_opencv_reference():
    import cv2

    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    intrinsics = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]])
    coefficients = np.array([0.1, -0.05, 0.001, 0.002, 0.0])
    corrected, output_intrinsics, lens_transform = da3_geometry._undistort(
        image,
        {
            "intrinsics": intrinsics.tolist(),
            "distortion": {
                "model": "opencv-brown-conrady",
                "coefficients": coefficients.tolist(),
            },
        },
    )
    expected_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics, coefficients, (64, 48), 0.0, (64, 48)
    )
    expected = cv2.undistort(image, intrinsics, coefficients, None, expected_intrinsics)
    source_mask = rng.integers(0, 2, (48, 64), dtype=np.uint8)
    processed_mask = da3_geometry._mask_to_processed(
        source_mask,
        width=64,
        height=48,
        scale_x=1.0,
        scale_y=1.0,
        lens_transform=lens_transform,
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        intrinsics,
        coefficients,
        np.eye(3),
        expected_intrinsics,
        (64, 48),
        cv2.CV_32FC1,
    )
    expected_mask = (
        cv2.remap(
            source_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
        > 0
    )

    assert output_intrinsics == pytest.approx(expected_intrinsics)
    assert np.array_equal(corrected, expected)
    assert np.array_equal(processed_mask, expected_mask)


def test_skew_without_distortion_keeps_source_raster_and_intrinsics():
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    intrinsics = [[50.0, 3.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
    corrected, output_intrinsics, lens = da3_geometry._undistort(
        image,
        {
            "intrinsics": intrinsics,
            "distortion": {"model": "none", "coefficients": []},
        },
    )

    assert corrected is image
    assert output_intrinsics == pytest.approx(np.asarray(intrinsics))
    assert lens == {"model": "none", "applied": False}


def test_nim_skeleton_bbox_is_projected_onto_native_depth_and_excluded(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    request_filter = {
        "excluded_pixel_regions": [
            {
                "frame_id": 0,
                "bbox": [20.0, 10.0, 40.0, 35.0],
                "source": "nim_skeleton",
            }
        ]
    }

    regions = da3_geometry._load_exclusion_masks(assets, request_filter)[0]
    source_mask = da3_geometry._source_exclusion_mask(regions, width=64, height=48)
    processed_mask = da3_geometry._mask_to_processed(
        source_mask,
        width=64,
        height=48,
        scale_x=56 / 64,
        scale_y=42 / 48,
        lens_transform={"applied": False},
    )
    depth = np.ones((42, 56), dtype=np.float32)
    filtered = da3_geometry._exclude_masked_depth(depth, processed_mask)

    assert processed_mask.shape == (42, 56)
    assert np.isnan(filtered[processed_mask]).all()
    assert np.isfinite(filtered[~processed_mask]).all()
    assert np.isfinite(depth).all()
