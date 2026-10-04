"""Offline-first, closed-set pose classification study tools.

This module is research tooling. It has no free-form retrieval or ranking command.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import tempfile
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

MODEL = "typesafe/jev-1.13"
ENDPOINT = "https://openrouter.ai/api/v1/systemone"
MANIFEST = Path(__file__).with_name("manifest.toml")
RESULT_FILES = (
    "action_quality_v1.jsonl",
    "quality_generic_v1.jsonl",
    "injected_corruption_v1.jsonl",
    "spec_assist_v1.jsonl",
)
JSON_FILES = ("windows.json", "source.json")

ACTION_CLASSES = {
    "other_or_ambiguous": "No single listed motion pattern clearly dominates.",
    "mostly_still": "The tracked body is mostly still.",
    "repetitive_upper_body": "The upper body shows repeated motion.",
    "deep_bend": "The body shows a deep bend or squat.",
    "large_root_motion": "The body root moves substantially through space.",
    "arms_overhead": "Both arms are raised above the head.",
}

FIXED_NOULS = {
    "action_deep_bend": (
        "Are at least half of observed knee angles below 120 degrees? Use the supplied fraction."
    ),
    "action_arms_overhead": (
        "Are both wrists above HeadTop in at least half of observed frames? Use the supplied fraction."
    ),
    "action_repetitive_upper": (
        "Does one wrist have at least 2 vertical direction reversals and a range of 0.5 body scales?"
    ),
    "action_large_root_motion": (
        "Does maximum root speed exceed 0.7 body scales per second or vertical range exceed 0.25?"
    ),
    "action_mostly_still": (
        "Is median root speed below 0.1 and median joint speed below 0.12 body scales per second?"
    ),
    "generic_unreliable": "Is this pose window unreliable for interpreting its movement?",
}

POSE_JOINTS = (
    "Hips",
    "HeadTop",
    "Chest",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
)


class ResearchError(ValueError):
    """Raised when research input or retained evidence fails validation."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ResearchError(f"{label} must be a finite number")
    return number


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ResearchError(f"value is not JSON-safe: {error}") from error
    return _sha256_bytes(encoded.encode("utf-8"))


def summarize_pose(pose: dict[str, Any]) -> dict[str, Any]:
    """Summarize one synthetic or local pose window with explicit axis handling.

    Required fields: source_frames, fps, coordinate_convention, and a joints map
    whose values are one 3-vector per source frame; a missing joint is ``None``.
    The body scale is the median valid Hips-to-HeadTop length. Root height is
    centered on the window median, never relative to itself frame by frame.
    """
    if not isinstance(pose, dict):
        raise ResearchError("pose must be an object")
    frames = pose.get("source_frames")
    if (
        not isinstance(frames, list)
        or len(frames) < 2
        or any(type(frame) is not int or frame < 0 for frame in frames)
        or any(right <= left for left, right in itertools.pairwise(frames))
    ):
        raise ResearchError("source_frames must contain at least two strictly increasing integers")
    fps = _finite_number(pose.get("fps"), "fps")
    if fps <= 0:
        raise ResearchError("fps must be positive")
    convention = pose.get("coordinate_convention")
    if convention not in {"raw_camera_y_down", "stage_y_up"}:
        raise ResearchError("coordinate_convention must be raw_camera_y_down or stage_y_up")
    y_sign = -1.0 if convention == "raw_camera_y_down" else 1.0
    joints = pose.get("joints")
    if not isinstance(joints, dict):
        raise ResearchError("joints must be an object")

    values: dict[str, list[tuple[float, float, float] | None]] = {}
    for name in POSE_JOINTS:
        rows = joints.get(name)
        if rows is None:
            values[name] = [None] * len(frames)
            continue
        if not isinstance(rows, list) or len(rows) != len(frames):
            raise ResearchError(f"joints.{name} must have one vector or null per source frame")
        parsed: list[tuple[float, float, float] | None] = []
        for index, row in enumerate(rows):
            if row is None:
                parsed.append(None)
                continue
            if not isinstance(row, list) or len(row) != 3:
                raise ResearchError(f"joints.{name}[{index}] must be a 3-number vector or null")
            parsed.append(
                (
                    _finite_number(row[0], f"joints.{name}[{index}]"),
                    _finite_number(row[1], f"joints.{name}[{index}]"),
                    _finite_number(row[2], f"joints.{name}[{index}]"),
                )
            )
        values[name] = parsed

    hips = values["Hips"]
    heads = values["HeadTop"]
    scales = [
        math.dist(hip, head)
        for hip, head in zip(hips, heads, strict=True)
        if hip is not None and head is not None
    ]
    if not scales:
        raise ResearchError(
            "at least one frame needs both Hips and HeadTop to establish body scale"
        )
    body_scale = statistics.median(scales)
    if body_scale <= 0:
        raise ResearchError("median Hips-to-HeadTop body scale must be positive")
    valid_hips = [(index, hip) for index, hip in enumerate(hips) if hip is not None]
    hip_y_center = statistics.median(y_sign * hip[1] for _, hip in valid_hips)
    relative_hip_y = [
        None if hip is None else (y_sign * hip[1] - hip_y_center) / body_scale for hip in hips
    ]

    root_speeds: list[float] = []
    joint_speeds: list[float] = []
    joint_speed_possible = (len(frames) - 1) * len(POSE_JOINTS)
    joint_speed_valid = 0
    gaps = 0
    largest_gap = 0
    for index, (left, right) in enumerate(itertools.pairwise(frames)):
        frame_delta = right - left
        if frame_delta > 1:
            gaps += 1
            largest_gap = max(largest_gap, frame_delta)
        hip_a, hip_b = hips[index], hips[index + 1]
        if frame_delta == 1 and hip_a is not None and hip_b is not None:
            distance = math.dist(hip_a, hip_b)
            root_speeds.append(distance * fps / body_scale)
        if frame_delta == 1:
            for rows in values.values():
                joint_a, joint_b = rows[index], rows[index + 1]
                if joint_a is not None and joint_b is not None:
                    joint_speeds.append(math.dist(joint_a, joint_b) * fps / body_scale)
                    joint_speed_valid += 1

    def coverage(valid_count: int, total_count: int) -> dict[str, float | int]:
        return {
            "valid_samples": valid_count,
            "total_samples": total_count,
            "fraction": valid_count / total_count if total_count else 0.0,
        }

    knee_angles: list[float] = []
    for side in ("L", "R"):
        for hip, knee, ankle in zip(
            values[f"{side}_Hip"], values[f"{side}_Knee"], values[f"{side}_Ankle"], strict=True
        ):
            if hip is None or knee is None or ankle is None:
                continue
            upper = tuple(hip[axis] - knee[axis] for axis in range(3))
            lower = tuple(ankle[axis] - knee[axis] for axis in range(3))
            upper_length = math.sqrt(sum(value * value for value in upper))
            lower_length = math.sqrt(sum(value * value for value in lower))
            if upper_length == 0 or lower_length == 0:
                continue
            cosine = sum(a * b for a, b in zip(upper, lower, strict=True)) / (
                upper_length * lower_length
            )
            knee_angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))

    overhead_frames = 0
    overhead_valid = 0
    wrist_ranges: dict[str, float | None] = {}
    wrist_reversals: dict[str, int | None] = {}
    wrist_valid_pairs = 0
    wrist_possible_pairs = len(frames) - 1
    for index, head in enumerate(values["HeadTop"]):
        left, right = values["L_Wrist"][index], values["R_Wrist"][index]
        if head is not None and left is not None and right is not None:
            overhead_valid += 1
            if y_sign * left[1] > y_sign * head[1] and y_sign * right[1] > y_sign * head[1]:
                overhead_frames += 1
    for side in ("L", "R"):
        wrist_rows = values[f"{side}_Wrist"]
        wrist_y = [
            None if wrist is None else y_sign * wrist[1] / body_scale for wrist in wrist_rows
        ]
        observed = [value for value in wrist_y if value is not None]
        wrist_ranges[side] = max(observed) - min(observed) if observed else None
        reversals = 0
        previous_direction: int | None = None
        wrist_pair_count = 0
        for index, (left_frame, right_frame) in enumerate(itertools.pairwise(frames)):
            left_y, right_y = wrist_y[index], wrist_y[index + 1]
            if right_frame - left_frame != 1 or left_y is None or right_y is None:
                previous_direction = None
                continue
            wrist_valid_pairs += 1
            wrist_pair_count += 1
            delta = right_y - left_y
            if abs(delta) >= 0.08:
                direction = 1 if delta > 0 else -1
                if previous_direction is not None and direction != previous_direction:
                    reversals += 1
                previous_direction = direction
        wrist_reversals[side] = reversals if wrist_pair_count else None

    missing_by_joint = {
        name: sum(row is None for row in rows) / len(frames) for name, rows in values.items()
    }
    return {
        "schema_version": "pose-summary-1",
        "coordinate_convention": convention,
        "source_frame_count": len(frames),
        "body_scale_headtop_to_hips_median": body_scale,
        "hip_y_body_scales_from_window_median": relative_hip_y,
        "root_speed_body_scales_s": root_speeds,
        "root_speed_median_body_scales_s": statistics.median(root_speeds) if root_speeds else None,
        "root_speed_max_body_scales_s": max(root_speeds) if root_speeds else None,
        "hip_vertical_range_body_scales": max(
            value for value in relative_hip_y if value is not None
        )
        - min(value for value in relative_hip_y if value is not None),
        "knee_angle_median_deg": statistics.median(knee_angles) if knee_angles else None,
        "knee_bend_fraction_below_120_deg": sum(angle < 120 for angle in knee_angles)
        / len(knee_angles)
        if knee_angles
        else None,
        "both_wrists_above_head_fraction": overhead_frames / overhead_valid
        if overhead_valid
        else None,
        "wrist_vertical_range_body_scales_by_side": wrist_ranges,
        "wrist_vertical_reversals_by_side": wrist_reversals,
        "joint_speed_median_body_scales_s": statistics.median(joint_speeds)
        if joint_speeds
        else None,
        "joint_speed_p90_body_scales_s": statistics.quantiles(joint_speeds, n=10)[8]
        if len(joint_speeds) >= 2
        else (joint_speeds[0] if joint_speeds else None),
        "source_frame_gaps": {"count": gaps, "largest": largest_gap},
        "missing_fraction_by_joint": missing_by_joint,
        "evidence_coverage": {
            "root_speed": coverage(len(root_speeds), len(frames) - 1),
            "knee_angle": coverage(len(knee_angles), len(frames) * 2),
            "both_wrists_above_head": coverage(overhead_valid, len(frames)),
            "wrist_vertical_trajectory": coverage(wrist_valid_pairs, wrist_possible_pairs * 2),
            "joint_speed": coverage(joint_speed_valid, joint_speed_possible),
        },
    }


def build_request(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed action and pose-quality request; no user query is accepted."""
    validate_pose_summary(summary)
    return {
        "model": MODEL,
        "state": {
            "candidate": {"digest": summary},
            "task": "Classify only the numeric pose-derived features in candidate.digest.",
        },
        "questions": {
            "action_stage": {
                "type": "choice",
                "instructions": (
                    "Which listed movement pattern best fits the numeric action features in "
                    "candidate.digest? Treat null features as unavailable; select "
                    "other_or_ambiguous when no observed feature supports a specific label. "
                    "Do not infer missing measurements."
                ),
                "criteria": ACTION_CLASSES,
            },
            **{
                question_id: {
                    "type": "noul",
                    "instructions": {
                        "question": question,
                        "evidence": "candidate.digest",
                        "rule": "If the named metric is null or has insufficient coverage, answer false.",
                    },
                    "criteria": {
                        "true": "The numeric evidence supports the described behavior.",
                        "false": "The evidence does not support it or is insufficient.",
                    },
                }
                for question_id, question in FIXED_NOULS.items()
            },
        },
    }


def validate_pose_summary(summary: Any) -> None:
    required_features = {
        "schema_version",
        "coordinate_convention",
        "source_frame_count",
        "body_scale_headtop_to_hips_median",
        "hip_y_body_scales_from_window_median",
        "root_speed_body_scales_s",
        "root_speed_median_body_scales_s",
        "root_speed_max_body_scales_s",
        "hip_vertical_range_body_scales",
        "knee_angle_median_deg",
        "knee_bend_fraction_below_120_deg",
        "both_wrists_above_head_fraction",
        "wrist_vertical_range_body_scales_by_side",
        "wrist_vertical_reversals_by_side",
        "joint_speed_median_body_scales_s",
        "joint_speed_p90_body_scales_s",
        "source_frame_gaps",
        "missing_fraction_by_joint",
        "evidence_coverage",
    }
    if not isinstance(summary, dict) or not required_features <= summary.keys():
        missing = (
            required_features - summary.keys() if isinstance(summary, dict) else required_features
        )
        raise ResearchError(
            f"pose summary pose-summary-1 is missing required features: {sorted(missing)}"
        )
    if not isinstance(summary, dict) or summary.get("schema_version") != "pose-summary-1":
        raise ResearchError("pose summary must use schema_version pose-summary-1")
    if summary.get("coordinate_convention") not in {"raw_camera_y_down", "stage_y_up"}:
        raise ResearchError("pose summary has an unknown coordinate convention")
    if type(summary.get("source_frame_count")) is not int or summary["source_frame_count"] < 2:
        raise ResearchError("pose summary source_frame_count must be an integer >= 2")
    scale = _finite_number(summary.get("body_scale_headtop_to_hips_median"), "body scale")
    if scale <= 0:
        raise ResearchError("body scale must be positive")
    heights = summary.get("hip_y_body_scales_from_window_median")
    if not isinstance(heights, list) or len(heights) != summary["source_frame_count"]:
        raise ResearchError("hip height feature must have one entry per source frame")
    for value in heights:
        if value is not None:
            _finite_number(value, "hip height feature")
    speeds = summary.get("root_speed_body_scales_s")
    if not isinstance(speeds, list):
        raise ResearchError("root speeds must be an array")
    for speed in speeds:
        if _finite_number(speed, "root speed") < 0:
            raise ResearchError("root speeds cannot be negative")
    gaps = summary.get("source_frame_gaps")
    if (
        not isinstance(gaps, dict)
        or type(gaps.get("count")) is not int
        or type(gaps.get("largest")) is not int
        or gaps["count"] < 0
        or gaps["largest"] < 0
    ):
        raise ResearchError("source_frame_gaps must contain nonnegative integer count/largest")
    speeds = summary["root_speed_body_scales_s"]
    median_speed = summary.get("root_speed_median_body_scales_s")
    if speeds:
        if _finite_number(median_speed, "median root speed") != statistics.median(speeds):
            raise ResearchError("median root speed does not match the root speed observations")
    elif median_speed is not None:
        raise ResearchError("median root speed must be null when no adjacent root speeds exist")
    max_speed = summary.get("root_speed_max_body_scales_s")
    if speeds:
        if _finite_number(max_speed, "maximum root speed") != max(speeds):
            raise ResearchError("maximum root speed does not match the root speed observations")
    elif max_speed is not None:
        raise ResearchError("maximum root speed must be null when no adjacent root speeds exist")
    missing = summary.get("missing_fraction_by_joint")
    if not isinstance(missing, dict) or set(missing) != set(POSE_JOINTS):
        raise ResearchError("missing_fraction_by_joint must cover every canonical pose joint")
    for name, value in missing.items():
        fraction = _finite_number(value, f"missing fraction for {name}")
        if not 0 <= fraction <= 1:
            raise ResearchError(f"missing fraction for {name} must be between 0 and 1")
    feature_coverage = summary.get("evidence_coverage")
    coverage_names = {
        "root_speed",
        "knee_angle",
        "both_wrists_above_head",
        "wrist_vertical_trajectory",
        "joint_speed",
    }
    if not isinstance(feature_coverage, dict) or set(feature_coverage) != coverage_names:
        raise ResearchError("evidence_coverage must include every fixed action measurement")
    for name, record in feature_coverage.items():
        if not isinstance(record, dict) or set(record) != {
            "valid_samples",
            "total_samples",
            "fraction",
        }:
            raise ResearchError(f"evidence coverage for {name} has an invalid shape")
        valid_count, total_count = record["valid_samples"], record["total_samples"]
        fraction = _finite_number(record["fraction"], f"coverage fraction for {name}")
        if (
            type(valid_count) is not int
            or type(total_count) is not int
            or valid_count < 0
            or total_count <= 0
            or valid_count > total_count
            or not 0 <= fraction <= 1
            or not math.isclose(fraction, valid_count / total_count, rel_tol=0, abs_tol=1e-12)
        ):
            raise ResearchError(f"evidence coverage for {name} has invalid counts")
    for field in (
        "hip_vertical_range_body_scales",
        "joint_speed_median_body_scales_s",
        "joint_speed_p90_body_scales_s",
    ):
        raw_value = summary.get(field)
        if raw_value is None and field != "hip_vertical_range_body_scales":
            continue
        value = _finite_number(raw_value, field)
        if value < 0:
            raise ResearchError(f"{field} cannot be negative")
    for field in (
        "knee_angle_median_deg",
        "knee_bend_fraction_below_120_deg",
        "both_wrists_above_head_fraction",
    ):
        value = summary.get(field)
        if value is not None:
            number = _finite_number(value, field)
            if field == "knee_angle_median_deg" and not 0 <= number <= 180:
                raise ResearchError(f"{field} must be between 0 and 180 degrees")
            if (field.endswith("_fraction_below_120_deg") or field.endswith("_fraction")) and not (
                0 <= number <= 1
            ):
                raise ResearchError(f"{field} must be between 0 and 1")
    for field in ("wrist_vertical_range_body_scales_by_side", "wrist_vertical_reversals_by_side"):
        values = summary.get(field)
        if not isinstance(values, dict) or set(values) != {"L", "R"}:
            raise ResearchError(f"{field} must have left and right measurements")
        for value in values.values():
            if value is not None:
                number = _finite_number(value, field)
                if field == "wrist_vertical_range_body_scales_by_side" and number < 0:
                    raise ResearchError(f"{field} cannot be negative")
                if field == "wrist_vertical_reversals_by_side" and (
                    type(value) is not int or value < 0
                ):
                    raise ResearchError(
                        f"{field} values must be nonnegative integer counts or null"
                    )


def validate_request(request: Any) -> None:
    """Validate TypeSafe System One request structure before any I/O."""
    if not isinstance(request, dict) or set(request) != {"model", "state", "questions"}:
        raise ResearchError("request must contain exactly model, state, and questions")
    if not isinstance(request["model"], str) or not request["model"].strip():
        raise ResearchError("request.model must be a nonempty string")
    if not isinstance(request["state"], (str, dict, list)):
        raise ResearchError("request.state must be a string, object, or array")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ResearchError("request.questions must be a nonempty object")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ResearchError("question ids must be nonempty strings")
        if not isinstance(question, dict) or not isinstance(
            question.get("instructions"), (str, dict, list)
        ):
            raise ResearchError(f"question {question_id} must have typed instructions")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ResearchError(f"choice question {question_id} needs 1 to 255 criteria")
        elif kind == "noul":
            if criteria is not None and not isinstance(criteria, dict):
                raise ResearchError(f"noul question {question_id} criteria must be an object")
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ResearchError(f"score question {question_id} needs 2 to 10 levels")
        else:
            raise ResearchError(f"question {question_id} has unsupported type {kind!r}")


def validate_response(response: Any, request: dict[str, Any]) -> None:
    """Validate provider response against the exact typed questions sent."""
    validate_request(request)
    if not isinstance(response, dict):
        raise ResearchError("response must be an object")
    if not isinstance(response.get("model"), str) or not response["model"]:
        raise ResearchError("response.model must be a nonempty string")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(request["questions"]):
        raise ResearchError("response.answers must contain exactly the requested question ids")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ResearchError("response.usage must be an object")
    for name in ("input_tokens", "output_tokens"):
        value = usage.get(name)
        if type(value) is not int or value < 0:
            raise ResearchError(f"response.usage.{name} must be a nonnegative integer")
    if "cost" in usage and _finite_number(usage["cost"], "response.usage.cost") < 0:
        raise ResearchError("response.usage.cost cannot be negative")

    for question_id, question in request["questions"].items():
        answer = answers[question_id]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ResearchError(f"answer {question_id} must match its requested type")
        kind = question["type"]
        if kind == "noul":
            value = _finite_number(answer.get("noul"), f"answer {question_id}.noul")
            if not 0 <= value <= 1:
                raise ResearchError(f"answer {question_id}.noul must be between 0 and 1")
        elif kind == "choice":
            criteria = question["criteria"]
            if answer.get("choice") not in criteria:
                raise ResearchError(f"answer {question_id}.choice is outside its criteria")
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
                raise ResearchError(
                    f"answer {question_id}.probabilities must cover every criterion"
                )
            numbers = [
                _finite_number(v, f"answer {question_id}.probabilities")
                for v in probabilities.values()
            ]
            if any(not 0 <= value <= 1 for value in numbers) or not math.isclose(
                sum(numbers), 1.0, rel_tol=0, abs_tol=0.002
            ):
                raise ResearchError(
                    f"answer {question_id}.probabilities must be in [0,1] and sum to 1"
                )
            confidence = _finite_number(
                answer.get("confidence"), f"answer {question_id}.confidence"
            )
            if not 0 <= confidence <= 1:
                raise ResearchError(f"answer {question_id}.confidence must be between 0 and 1")
        else:
            _finite_number(answer.get("score"), f"answer {question_id}.score")
            probabilities = answer.get("probabilities")
            legend = answer.get("legend")
            if not isinstance(legend, dict) or not isinstance(probabilities, dict):
                raise ResearchError(f"answer {question_id}.probabilities must be an object")
            expected_levels = {str(index) for index in range(len(question["criteria"]))}
            if set(legend) != expected_levels or not all(
                isinstance(value, str) for value in legend.values()
            ):
                raise ResearchError(f"answer {question_id}.legend must describe every score level")
            if set(probabilities) != expected_levels:
                raise ResearchError(
                    f"answer {question_id}.probabilities must cover every score level"
                )
            values = [
                _finite_number(v, f"answer {question_id}.probabilities")
                for v in probabilities.values()
            ]
            if (
                not values
                or any(not 0 <= value <= 1 for value in values)
                or not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=0.002)
            ):
                raise ResearchError(f"answer {question_id}.probabilities must sum to 1")
            confidence = _finite_number(
                answer.get("confidence"), f"answer {question_id}.confidence"
            )
            if not 0 <= confidence <= 1:
                raise ResearchError(f"answer {question_id}.confidence must be between 0 and 1")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchError(f"cannot read valid JSON from {path}: {error}") from error


def _manifest_data() -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef,import-not-found]
    try:
        return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResearchError(f"cannot load artifact manifest: {error}") from error


def verify_processing_sources(manifest: dict[str, Any]) -> None:
    processing_sources = manifest.get("processing_sources")
    if not isinstance(processing_sources, dict) or set(processing_sources) != {
        "frozen_stage.py",
        "frozen_nova77.py",
        "frozen_processing.py",
        "nova77.py",
    }:
        raise ResearchError("manifest must pin the four local frozen processing sources")
    for name, metadata in processing_sources.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            raise ResearchError(f"processing source {name} needs a sha256")
        path = MANIFEST.parent / name
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise ResearchError(f"missing frozen processing source {path}: {error}") from error
        if actual != metadata["sha256"]:
            raise ResearchError(f"frozen processing source checksum mismatch: {name}")


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ResearchError(f"cannot read {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResearchError(f"{path.name}:{line_number} is not valid JSON: {error}") from error
        if not isinstance(row, dict) or row.get("status") != "ok":
            raise ResearchError(f"{path.name}:{line_number} is not a successful result record")
        payload = {
            "model": row.get("model_requested"),
            "state": row.get("state"),
            "questions": row.get("questions"),
        }
        validate_request(payload)
        if row.get("request_sha256") != canonical_hash(payload):
            raise ResearchError(
                f"{path.name}:{line_number} request hash does not match its payload"
            )
        if row.get("state_sha256") != canonical_hash(payload["state"]):
            raise ResearchError(f"{path.name}:{line_number} state hash does not match its payload")
        if row.get("questions_sha256") != canonical_hash(payload["questions"]):
            raise ResearchError(
                f"{path.name}:{line_number} questions hash does not match its payload"
            )
        if row.get("question_count") != len(payload["questions"]):
            raise ResearchError(
                f"{path.name}:{line_number} question count does not match its payload"
            )
        validate_response(row.get("response"), payload)
        rows.append(row)
    if not rows:
        raise ResearchError(f"{path.name} has no result records")
    return rows


def verify_artifacts(directory: Path) -> dict[str, int]:
    """Verify all tracked checksums and the retained TypeSafe result shapes."""
    manifest = _manifest_data()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ResearchError("manifest must define an artifacts table")
    expected = set(JSON_FILES + RESULT_FILES)
    if set(artifacts) != expected:
        raise ResearchError("manifest artifact names must match the six retained input files")
    counts: dict[str, int] = {}
    verify_processing_sources(manifest)
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str):
            raise ResearchError(f"manifest entry {name} needs a sha256")
        path = directory / name
        try:
            actual = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise ResearchError(f"missing artifact {path}: {error}") from error
        if actual != metadata["sha256"]:
            raise ResearchError(f"artifact checksum mismatch: {name}")
        if name.endswith(".jsonl"):
            counts[name] = len(_jsonl_rows(path))
        else:
            document = _load_json(path)
            if not isinstance(document, dict):
                raise ResearchError(f"{name} must contain a JSON object")
            counts[name] = 1
    return counts


def summarize_artifacts(directory: Path) -> dict[str, Any]:
    """Return fixed counts over named classification files; there is no query filter."""
    counts = verify_artifacts(directory)
    labels: dict[str, dict[str, dict[str, int]]] = {}
    for filename in ("action_quality_v1.jsonl", "spec_assist_v1.jsonl"):
        labels[filename] = {}
    usage = {"requests": 0, "judgments": 0, "input_tokens": 0, "output_tokens": 0}
    provider_cost = Decimal("0")
    complete_cost = True
    for filename in RESULT_FILES:
        for row in _jsonl_rows(directory / filename):
            response_usage = row["response"]["usage"]
            usage["requests"] += 1
            usage["judgments"] += row["question_count"]
            usage["input_tokens"] += response_usage["input_tokens"]
            usage["output_tokens"] += response_usage["output_tokens"]
            if "cost" in response_usage:
                provider_cost += Decimal(str(response_usage["cost"]))
            else:
                complete_cost = False
            if filename in labels:
                for question_id, answer in row["response"]["answers"].items():
                    if answer["type"] == "choice":
                        label = answer["choice"]
                        label_counts = labels[filename].setdefault(question_id, {})
                        label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "schema_version": "classification-summary-1",
        "records": counts,
        "choice_labels": labels,
        "usage": {
            **usage,
            "provider_cost_usd": float(provider_cost) if complete_cost else None,
        },
    }


def _replay_rows(directory: Path) -> list[dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for filename in RESULT_FILES:
        for row in _jsonl_rows(directory / filename):
            request = {
                "model": row["model_requested"],
                "state": row["state"],
                "questions": row["questions"],
            }
            replay_rows.append(
                {
                    "experiment": row["experiment"],
                    "experiment_item_id": row.get("experiment_item_id"),
                    "request_sha256": row["request_sha256"],
                    "request": request,
                    "response": row["response"],
                }
            )
    return replay_rows


def replay_artifacts(directory: Path, output: Path) -> int:
    """Atomically export the exact retained TypeSafe requests and responses offline."""
    verify_artifacts(directory)
    replay_rows = _replay_rows(directory)
    encoded = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
        for row in replay_rows
    )
    _atomic_write(output, encoded)
    return len(replay_rows)


def build_frozen_stage(pose_file: Path, output: Path) -> None:
    """Run the exact pinned pose-to-stage processor and atomically save its payload."""
    manifest = _manifest_data()
    verify_processing_sources(manifest)
    from . import frozen_processing

    frames = frozen_processing.load_frames(str(pose_file))
    stage = frozen_processing.build_stage(frames, frozen_processing.Options(fps=30.0))
    result = {
        "generating_processing_commit": manifest["generating_processing_commit"],
        "meta": stage.meta,
        "payload": stage.payload(),
    }
    encoded = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _atomic_write(output, encoded)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_cache(
    path: Path, request: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not path.exists():
        return [], None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ResearchError(f"cannot read response cache: {error}") from error
    cached: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResearchError(f"cache line {line_number} is invalid JSON: {error}") from error
        if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
            raise ResearchError(f"cache line {line_number} is not a request/response record")
        validate_request(row["request"])
        digest = canonical_hash(row["request"])
        if row.get("request_sha256") != digest:
            raise ResearchError(f"cache line {line_number} has a bad request checksum")
        validate_response(row.get("response"), row["request"])
        rows.append(row)
        if digest == canonical_hash(request):
            cached = row
    return rows, cached


def classify(pose: dict[str, Any], cache_path: Path, output_path: Path) -> dict[str, Any]:
    """Classify one pose window, replaying exact cached requests without credentials."""
    summary = summarize_pose(pose)
    request = build_request(summary)
    request_hash = canonical_hash(request)
    if output_path.exists():
        previous = _load_json(output_path)
        if (
            not isinstance(previous, dict)
            or previous.get("schema_version") != "pose-classification-result-1"
            or previous.get("request_sha256") != request_hash
        ):
            raise ResearchError("existing output is not a result for this exact request")
        validate_pose_summary(previous.get("summary"))
        validate_response(previous.get("response"), request)
    old_rows, cached = _load_cache(cache_path, request)
    cache_hit = cached is not None
    if cached is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ResearchError(
                "OPENROUTER_API_KEY is required when no exact cached response exists"
            )
        body = json.dumps(request, separators=(",", ":"), allow_nan=False).encode("utf-8")
        req = urllib.request.Request(
            ENDPOINT,
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as response:  # noqa: S310 -- fixed HTTPS endpoint
                parsed = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise ResearchError(f"System One request failed: {error}") from error
        validate_response(parsed, request)
        cached = {"request_sha256": request_hash, "request": request, "response": parsed}
        new_cache = [*old_rows, cached]
        cache_bytes = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            for row in new_cache
        )
        _atomic_write(cache_path, cache_bytes)
    result = {
        "schema_version": "pose-classification-result-1",
        "request_sha256": request_hash,
        "summary": summary,
        "response": cached["response"],
        "cached": cache_hit,
    }
    validate_response(result["response"], request)
    _atomic_write(
        output_path, (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )
    return result


def synthetic_preflight() -> None:
    """Exercise stationary/bouncing geometry, gaps, missing joints, and wire schemas offline."""
    frames = [10, 11, 12, 14]
    stage = {
        "source_frames": frames,
        "fps": 30,
        "coordinate_convention": "stage_y_up",
        "joints": {
            "Hips": [[0, y, 0] for y in (0, 1, 0, -1)],
            "HeadTop": [[0, y + 2, 0] for y in (0, 1, 0, -1)],
            "L_Wrist": [[0, 2, 0], None, [0, 1, 0], [0, 0, 0]],
        },
    }
    summary = summarize_pose(stage)
    expected = [0.0, 0.5, 0.0, -0.5]
    if summary["hip_y_body_scales_from_window_median"] != expected:
        raise ResearchError(
            "hip feature does not preserve bouncing root movement around its median"
        )
    stationary = summarize_pose(
        {
            "source_frames": [0, 1, 2],
            "fps": 30,
            "coordinate_convention": "stage_y_up",
            "joints": {
                "Hips": [[0, 5, 0], [0, 5, 0], [0, 5, 0]],
                "HeadTop": [[0, 7, 0], [0, 7, 0], [0, 7, 0]],
            },
        }
    )
    if stationary["hip_y_body_scales_from_window_median"] != [0.0, 0.0, 0.0]:
        raise ResearchError("stationary root must have zero median-centered hip height")
    if stationary["root_speed_median_body_scales_s"] != 0:
        raise ResearchError("stationary root must have zero median speed")
    nonfinite_pose = {
        "source_frames": [0, 1],
        "fps": 30,
        "coordinate_convention": "stage_y_up",
        "joints": {"Hips": [[0, math.nan, 0], [0, 0, 0]], "HeadTop": [[0, 2, 0], [0, 2, 0]]},
    }
    try:
        summarize_pose(nonfinite_pose)
    except ResearchError:
        pass
    else:
        raise ResearchError("preflight accepted a non-finite pose measurement")
    if summary["source_frame_gaps"] != {"count": 1, "largest": 2}:
        raise ResearchError("frame gap was not detected")
    if summary["missing_fraction_by_joint"]["L_Wrist"] != 0.25:
        raise ResearchError("missing joint was not preserved as missing data")
    raw = dict(stage, coordinate_convention="raw_camera_y_down")
    raw_summary = summarize_pose(raw)
    if raw_summary["hip_y_body_scales_from_window_median"] != [0.0, -0.5, 0.0, 0.5]:
        raise ResearchError("camera-y-down normalization did not flip the vertical sign")
    request = build_request(summary)
    validate_request(request)
    validate_response(_fixture_response(request), request)
    bad_response = _fixture_response(request)
    bad_response["answers"]["action_stage"]["probabilities"] = {"unlisted": 1.0}
    try:
        validate_response(bad_response, request)
    except ResearchError:
        pass
    else:
        raise ResearchError("preflight accepted a malformed TypeSafe response")


def _fixture_response(request: dict[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for question_id, question in request["questions"].items():
        if question["type"] == "choice":
            first = next(iter(question["criteria"]))
            answers[question_id] = {
                "type": "choice",
                "choice": first,
                "probabilities": {
                    key: (1.0 if key == first else 0.0) for key in question["criteria"]
                },
                "confidence": 1.0,
            }
        else:
            answers[question_id] = {"type": "noul", "noul": 0.0}
    return {
        "model": "typesafe/jev-1.13-fixture",
        "answers": answers,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="run synthetic geometry and API schema checks offline")
    verify_parser = subparsers.add_parser(
        "verify", help="check retained artifact hashes and schemas"
    )
    verify_parser.add_argument("--artifacts", type=Path, required=True)
    summary_parser = subparsers.add_parser(
        "summary", help="summarize fixed retained classification files"
    )
    summary_parser.add_argument("--artifacts", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path)
    replay_parser = subparsers.add_parser(
        "replay", help="export verified retained requests and responses offline"
    )
    replay_parser.add_argument("--artifacts", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    stage_parser = subparsers.add_parser(
        "frozen-stage", help="process local pose.json lines with the pinned study processor"
    )
    stage_parser.add_argument("--pose-json", type=Path, required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    classify_parser = subparsers.add_parser("classify", help="classify one numeric pose window")
    classify_parser.add_argument("--pose", type=Path, required=True)
    classify_parser.add_argument("--cache", type=Path, required=True)
    classify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            synthetic_preflight()
            print("preflight passed (offline synthetic pose and TypeSafe schema)")
        elif args.command == "verify":
            print(json.dumps(verify_artifacts(args.artifacts), sort_keys=True, indent=2))
        elif args.command == "summary":
            summary = summarize_artifacts(args.artifacts)
            encoded = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
            if args.output:
                _atomic_write(args.output, encoded.encode())
            else:
                print(encoded, end="")
        elif args.command == "replay":
            count = replay_artifacts(args.artifacts, args.output)
            print(f"replayed {count} verified request/response pairs to {args.output}")
        elif args.command == "frozen-stage":
            build_frozen_stage(args.pose_json, args.output)
            print(f"frozen stage written atomically to {args.output}")
        else:
            pose = _load_json(args.pose)
            if not isinstance(pose, dict):
                raise ResearchError("pose input must be a JSON object")
            classify(pose, args.cache, args.output)
            print(f"classification written atomically to {args.output}")
    except ResearchError as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
