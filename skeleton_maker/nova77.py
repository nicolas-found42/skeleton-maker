# SPDX-License-Identifier: MIT
"""Anatomy of the Nova-77 skeleton: what each joint index *is*.

``constants.NOVA77_PARENTS`` gives the topology. This module names the joints a
character needs. The names are inferred from NVIDIA's skeleton diagram
(nim-clients/body-pose/assets/nova77_skeleton.png) and checked against real NIM
output: bone lengths in ``rest_pose`` match the 3D keypoints, the clavicle ->
shoulder -> elbow -> wrist and hip -> knee -> ankle chains have anatomical
lengths, and toes point away from the heel. NVIDIA does not publish joint names,
so treat the names as ours and the indices as theirs.

The diagram shows two mirrored sides. Which one is the person's *left* is not
stated, so it is called side ``A`` (indices 11-38, 67-71, 9) and side ``B``
(indices 39-66, 72-76, 8) here and resolved from the data per tracked body in
:mod:`skeleton_maker.stage`.
"""

# Side A / side B source indices for each canonical limb joint.
SIDE_A = {"Clavicle": 11, "Shoulder": 12, "Elbow": 13, "Wrist": 14,
          "Hip": 67, "Knee": 68, "Ankle": 69, "Heel": 70, "Toe": 71, "HeadSide": 9,
          "FingerBases": (15, 19, 24, 29, 34), "HandTip": 28}
SIDE_B = {"Clavicle": 39, "Shoulder": 40, "Elbow": 41, "Wrist": 42,
          "Hip": 72, "Knee": 73, "Ankle": 74, "Heel": 75, "Toe": 76, "HeadSide": 8,
          "FingerBases": (43, 47, 52, 57, 62), "HandTip": 56}

# Joints on the midline, same for every body.
MIDLINE = {"Hips": 0, "Spine1": 1, "Spine2": 2, "Chest": 3, "Neck1": 4, "Neck2": 5,
           "Head": 6, "HeadTop": 7, "Face": 10}

#: Canonical joints a character is built from, in the order they are packed.
#: ``L_*`` and ``R_*`` are the person's own left and right. ``HandC`` is the palm
#: centre (mean of the five finger bases), ``HandTip`` the middle fingertip.
CANON = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadTop", "Face",
    "L_HeadSide", "R_HeadSide",
    "L_Clavicle", "L_Shoulder", "L_Elbow", "L_Wrist", "L_HandC", "L_HandTip",
    "R_Clavicle", "R_Shoulder", "R_Elbow", "R_Wrist", "R_HandC", "R_HandTip",
    "L_Hip", "L_Knee", "L_Ankle", "L_Heel", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Heel", "R_Toe",
)
CANON_INDEX = {name: i for i, name in enumerate(CANON)}


def canon_sources(a_is_left: bool) -> list:
    """For each canonical joint, the source index (or tuple of indices to average).

    ``a_is_left`` says whether side A is the person's left.
    """
    left, right = (SIDE_A, SIDE_B) if a_is_left else (SIDE_B, SIDE_A)
    out = []
    for name in CANON:
        if name in MIDLINE:
            out.append(MIDLINE[name])
            continue
        side, key = (left if name.startswith("L_") else right), name[2:]
        if key == "HandC":
            out.append(tuple(side["FingerBases"]))
        else:
            out.append(side[key])
    return out
