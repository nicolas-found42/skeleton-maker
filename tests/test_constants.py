# SPDX-License-Identifier: MIT
"""The Nova-77 topology is data; a wrong entry silently draws a wrong skeleton."""

from skeleton_maker.constants import (
    DRAW_KEYPOINTS_CONFIGS,
    NOVA77_PARENTS,
    NOVA77_SKELETON_LINKS,
    NUM_JOINTS,
    TRACK_COLORS,
)


def test_topology_has_one_parent_per_joint():
    assert len(NOVA77_PARENTS) == NUM_JOINTS


def test_exactly_one_root():
    roots = [i for i, p in enumerate(NOVA77_PARENTS) if p < 0]
    assert roots == [0], "joint 0 (Hips) must be the only root"


def test_every_parent_is_a_valid_earlier_or_any_joint():
    for child, parent in enumerate(NOVA77_PARENTS):
        if parent >= 0:
            assert 0 <= parent < NUM_JOINTS
            assert parent != child, f"joint {child} cannot parent itself"


def test_the_tree_is_acyclic():
    for start in range(NUM_JOINTS):
        seen, node = set(), start
        while node >= 0:
            assert node not in seen, f"cycle reachable from joint {start}"
            seen.add(node)
            node = NOVA77_PARENTS[node]


def test_skeleton_links_match_the_parents():
    expected = [(i, p) for i, p in enumerate(NOVA77_PARENTS) if p >= 0]
    assert expected == NOVA77_SKELETON_LINKS
    assert len(NOVA77_SKELETON_LINKS) == NUM_JOINTS - 1


def test_draw_configs_are_valid():
    assert set(DRAW_KEYPOINTS_CONFIGS) == {"2d", "3d", "both"}
    for flags in DRAW_KEYPOINTS_CONFIGS.values():
        assert len(flags) == 2
        assert any(flags)
    assert DRAW_KEYPOINTS_CONFIGS["2d"] == (True, False)
    assert DRAW_KEYPOINTS_CONFIGS["3d"] == (False, True)


def test_track_colors_are_bgr_triples():
    assert TRACK_COLORS, "need at least one color"
    for color in TRACK_COLORS:
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
