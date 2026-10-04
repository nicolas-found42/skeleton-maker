# SPDX-License-Identifier: MIT
"""Stage building on synthetic bodies with known geometry (no NIM, no network)."""

import numpy as np
import pytest

from skeleton_maker.nova77 import CANON, CANON_INDEX, canon_sources
from skeleton_maker.stage import Options, Stage, build_stage, detect_shots

# Joint positions of a 1.75 m person in a y-up world, facing +z, with side A (the
# higher-index diagram side, joints 9, 11-38, 67-71) on the person's left (+x).
_UP = {
    0: (0, .95, 0), 1: (0, 1.05, 0), 2: (0, 1.18, 0), 3: (0, 1.35, 0), 4: (0, 1.5, 0), 5: (0, 1.56, 0),
    6: (0, 1.62, 0), 7: (0, 1.75, 0), 10: (0, 1.66, .09), 9: (.043, 1.67, 0), 8: (-.043, 1.67, 0),
}


def _side(sign, clav, sh, el, wr, bases, tip, hip, knee, ank, heel, toe):
    d = {clav: (.05 * sign, 1.45, 0), sh: (.18 * sign, 1.45, 0), el: (.46 * sign, 1.45, 0), wr: (.72 * sign, 1.45, 0),
         tip: (.9 * sign, 1.45, 0), hip: (.09 * sign, .9, 0), knee: (.09 * sign, .5, 0), ank: (.09 * sign, .08, 0),
         heel: (.09 * sign, .02, -.05), toe: (.09 * sign, .02, .15)}
    for i, b in enumerate(bases):
        d[b] = (.78 * sign, 1.45, (i - 2) * .02)
    return d


_UP.update(_side(1, 11, 12, 13, 14, (15, 19, 24, 29, 34), 28, 67, 68, 69, 70, 71))
_UP.update(_side(-1, 39, 40, 41, 42, (43, 47, 52, 57, 62), 56, 72, 73, 74, 75, 76))


def body(origin=(0.0, 0.0, -5.0), facing=1.0, tilt_deg=0.0, conf=0.9, jitter=0.0, rng=None, mirror=False):
    """One detection dict. ``facing=-1`` turns the body to face away along z.

    ``mirror`` flips the lateral axis only, which puts side A on the person's right.
    """
    pts = np.tile(np.array(_UP[0], float), (77, 1))
    for i, p in _UP.items():
        pts[i] = p
    pts[:, 0] *= facing  # x and z together: a proper rotation of the body about y
    pts[:, 2] *= facing
    if mirror:
        pts[:, 0] *= -1
    pts = pts + np.array(origin)
    if jitter and rng is not None:
        pts = pts + rng.normal(0, jitter, pts.shape)
    t = np.radians(tilt_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    cam = (pts @ tilt.T) * np.array([1, -1, -1])  # y-up world -> OpenCV camera (y down, z forward)
    root = cam[0]
    return {"tracking_id": 1, "bbox": [0, 0, 1, 1], "keypoints_2d": np.zeros((77, 2)).tolist(),
            "keypoints_confidence": [conf] * 77, "keypoints_3d": cam.tolist(), "rest_pose": np.zeros((77, 3)).tolist(),
            "joint_rotations": [[0, 0, 0, 1]] * 77, "root_pose": {"translation": root.tolist(), "rotation": [0, 0, 0, 1]}}


def clip(n, **kw):
    return [(f, [dict(body(**kw))]) for f in range(n)]


def _first(stage, shot=0, track=0):
    return stage.positions(shot, track)


def test_canonical_joint_sources_cover_all_names():
    srcs = canon_sources(True)
    assert len(srcs) == len(CANON)
    flat = [i for s in srcs for i in (s if isinstance(s, tuple) else (s,))]
    assert all(0 <= i < 77 for i in flat)


def test_left_right_swap_when_sides_are_swapped():
    a, b = canon_sources(True), canon_sources(False)
    left = CANON_INDEX["L_Shoulder"]
    assert a[left] == 12 and b[left] == 40


def test_single_shot_single_track():
    st = build_stage(clip(40))
    assert len(st.meta["shots"]) == 1
    assert len(st.meta["shots"][0]["tracks"]) == 1
    p = _first(st)
    assert p.shape == (40, len(CANON), 3)


def test_person_facing_camera_has_side_a_as_left():
    st = build_stage(clip(40, facing=1.0))
    t = st.meta["shots"][0]["tracks"][0]
    assert t["sides_confident"] and t["a_is_left"]


def test_turning_the_body_around_does_not_change_which_side_is_left():
    st = build_stage(clip(40, facing=-1.0))
    t = st.meta["shots"][0]["tracks"][0]
    assert t["sides_confident"] and t["a_is_left"]


def test_a_body_whose_side_a_is_on_its_right_is_detected():
    st = build_stage(clip(40, mirror=True))
    t = st.meta["shots"][0]["tracks"][0]
    assert t["sides_confident"] and not t["a_is_left"]
    p = _first(st)
    # after canonicalising, the person's L_Shoulder must really be on their left (+x when facing +z)
    assert np.nanmedian(p[:, CANON_INDEX["L_Shoulder"], 0] - p[:, CANON_INDEX["R_Shoulder"], 0]) > 0.2


def test_tilted_camera_is_levelled_and_the_floor_is_zero():
    st = build_stage(clip(60, tilt_deg=12.0))
    p = _first(st)
    sole = np.nanmin(p[:, [CANON_INDEX["L_Heel"], CANON_INDEX["R_Toe"]], 1], axis=1)
    assert abs(np.nanmedian(sole)) < 0.05
    head = np.nanmedian(p[:, CANON_INDEX["HeadTop"], 1])
    assert head == pytest.approx(1.75 - 0.02, abs=0.08)  # heel/toe sit 2 cm above the true floor


def test_body_unit_matches_leg_length():
    st = build_stage(clip(30))
    assert st.meta["shots"][0]["tracks"][0]["unit"] == pytest.approx(1.0, abs=0.1)


def test_camera_cut_starts_a_new_shot():
    frames = clip(20, origin=(0, 0, -5.0))
    other = []
    for f in range(20, 40):
        d = body(origin=(2.0, 0, -9.0))
        d["tracking_id"] = 7  # tracker restarted
        other.append((f, [d]))
    shots = detect_shots(frames + other, Options())
    assert shots == [(0, 19), (20, 39)]


def test_kept_id_but_teleported_body_is_a_cut():
    frames = clip(10, origin=(0, 0, -5.0)) + [(f, [body(origin=(6.0, 0, -9.0))]) for f in range(10, 20)]
    assert len(detect_shots(frames, Options())) == 2


def test_long_absence_is_a_new_shot():
    frames = clip(10) + [(f, [body()]) for f in range(200, 210)]
    assert len(detect_shots(frames, Options())) == 2


def test_short_gap_is_interpolated_and_long_gap_ends_the_track():
    frames = [(f, [body()]) for f in range(40) if f not in (10, 11, 12)]
    st = build_stage(frames)
    p = _first(st)
    assert np.isfinite(p[10:13]).all(), "a 3-frame gap should be filled"
    gap = [(f, [body()]) for f in range(60) if not 10 <= f < 30]
    st = build_stage(gap, Options(min_len=6))
    assert len(st.meta["shots"][0]["tracks"]) == 2, "a 20-frame gap splits the track"


def test_low_confidence_joints_are_dropped_not_drawn():
    frames = []
    for f in range(30):
        d = body()
        d["keypoints_confidence"][69] = 0.0  # side A ankle never seen
        frames.append((f, [d]))
    st = build_stage(frames, Options(min_conf=0.05))
    p = _first(st)
    assert np.isnan(p[:, CANON_INDEX["L_Ankle"]]).all() or np.isnan(p[:, CANON_INDEX["R_Ankle"]]).all()


def test_frames_without_a_head_are_not_a_body():
    frames = []
    for f in range(30):
        d = body()
        d["keypoints_confidence"][6] = 0.0
        frames.append((f, [d]))
    assert build_stage(frames).meta["shots"] == []


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    frames = [(f, [body(jitter=0.01, rng=rng)]) for f in range(120)]
    raw = np.array([d["keypoints_3d"][3] for _, (d,) in frames])
    st = build_stage(frames)
    sm = _first(st)[:, CANON_INDEX["Chest"]]
    assert np.nanstd(np.diff(sm, axis=0)) < np.std(np.diff(raw, axis=0)) * 0.8


def test_payload_roundtrip_is_lossless_at_millimetre_precision():
    st = build_stage(clip(30))
    again = Stage.from_payload(st.payload())
    assert again.meta == st.meta
    np.testing.assert_allclose(again.positions(0, 0), st.positions(0, 0), atol=1e-9)


def test_far_people_do_not_overflow_the_int16_packing():
    st = build_stage(clip(30, origin=(0, 0, -45.0)))
    p = _first(st)
    assert np.isfinite(p).all()
    assert np.nanmedian(p[:, CANON_INDEX["Hips"], 2]) == pytest.approx(-45.0, abs=0.2)
