"""Offline safety checks for the pose classification research harness."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from experiments.pose_classification import research_battery as research


def pose_for_heights(
    heights: list[float], *, convention: str = "stage_y_up", frames: list[int] | None = None
) -> dict[str, Any]:
    return {
        "source_frames": frames or list(range(len(heights))),
        "fps": 30,
        "coordinate_convention": convention,
        "joints": {
            "Hips": [[0.0, height, 0.0] for height in heights],
            "HeadTop": [[0.0, height + 2.0, 0.0] for height in heights],
        },
    }


def complete_pose() -> dict[str, Any]:
    roots = [0.0, 1.0, 0.0, -1.0]
    offsets = {
        "Hips": (0.0, 0.0, 0.0),
        "HeadTop": (0.0, 2.0, 0.0),
        "Chest": (0.0, 1.0, 0.0),
        "L_Shoulder": (-0.5, 1.5, 0.0),
        "R_Shoulder": (0.5, 1.5, 0.0),
        "L_Elbow": (-0.5, 2.0, 0.0),
        "R_Elbow": (0.5, 2.0, 0.0),
        "L_Wrist": (-0.5, 3.0, 0.0),
        "R_Wrist": (0.5, 3.0, 0.0),
        "L_Hip": (-0.2, -0.1, 0.0),
        "R_Hip": (0.2, -0.1, 0.0),
        "L_Knee": (-0.2, -1.1, 0.0),
        "R_Knee": (0.2, -1.1, 0.0),
        "L_Ankle": (0.8, -1.1, 0.0),
        "R_Ankle": (0.8, -1.1, 0.0),
    }
    joints = {
        name: [[x, root + y, z] for root, x, y, z in ((root, *offset) for root in roots)]
        for name, offset in offsets.items()
    }
    joints["L_Wrist"] = [
        [-0.5, root + height, 0.0] for root, height in zip(roots, [3.0, 4.0, 3.0, 3.0], strict=True)
    ]
    return {
        "source_frames": [0, 1, 2, 3],
        "fps": 30,
        "coordinate_convention": "stage_y_up",
        "joints": joints,
    }


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=16,
    ),
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
)
def test_root_height_is_centered_on_window_median_and_normalized_by_body_scale(
    heights: list[float], translation: float
) -> None:
    moved = [height + translation for height in heights]
    summary = research.summarize_pose(pose_for_heights(moved))
    median = (
        sorted(moved)[len(moved) // 2]
        if len(moved) % 2
        else (sorted(moved)[len(moved) // 2 - 1] + sorted(moved)[len(moved) // 2]) / 2
    )
    expected = [(height - median) / 2 for height in moved]
    actual = summary["hip_y_body_scales_from_window_median"]
    assert actual == pytest.approx(expected)
    if max(heights) - min(heights) > 1e-8:
        assert any(abs(value) > 1e-8 for value in actual)


def test_camera_y_down_flips_vertical_feature_but_not_body_scale() -> None:
    stage = research.summarize_pose(pose_for_heights([0.0, 1.0, 0.0, -1.0]))
    camera = research.summarize_pose(
        pose_for_heights([0.0, 1.0, 0.0, -1.0], convention="raw_camera_y_down")
    )
    assert camera["hip_y_body_scales_from_window_median"] == pytest.approx(
        [-value for value in stage["hip_y_body_scales_from_window_median"]]
    )
    assert camera["body_scale_headtop_to_hips_median"] == pytest.approx(
        stage["body_scale_headtop_to_hips_median"]
    )


def test_stationary_root_is_zero_and_missing_joint_and_frame_gap_are_reported() -> None:
    pose = pose_for_heights([5.0, 5.0, 5.0], frames=[20, 21, 24])
    pose["joints"]["L_Wrist"] = [[0.0, 3.0, 0.0], None, [0.0, 3.0, 0.0]]
    summary = research.summarize_pose(pose)
    assert summary["hip_y_body_scales_from_window_median"] == [0.0, 0.0, 0.0]
    assert summary["root_speed_body_scales_s"] == [0.0]
    assert summary["source_frame_gaps"] == {"count": 1, "largest": 3}
    assert summary["missing_fraction_by_joint"]["L_Wrist"] == pytest.approx(1 / 3)
    assert summary["knee_bend_fraction_below_120_deg"] is None
    assert summary["evidence_coverage"]["knee_angle"]["fraction"] == 0
    assert summary["both_wrists_above_head_fraction"] is None


def test_fixed_action_measurements_use_present_joints_and_keep_coverage() -> None:
    summary = research.summarize_pose(complete_pose())
    assert summary["knee_bend_fraction_below_120_deg"] == pytest.approx(1)
    assert summary["both_wrists_above_head_fraction"] == pytest.approx(1)
    assert summary["wrist_vertical_reversals_by_side"]["L"] == 1
    assert summary["evidence_coverage"]["knee_angle"]["fraction"] == pytest.approx(1)
    request = research.build_request(summary)
    assert (
        "Treat null features as unavailable" in request["questions"]["action_stage"]["instructions"]
    )


@pytest.mark.parametrize(
    ("frames", "wrist_y"),
    [([0, 1, 3, 4], [3.0, 4.0, 2.0, 1.0]), ([0, 1, 2, 3], [3.0, 4.0, None, 2.0])],
)
def test_wrist_reversals_do_not_bridge_gaps_or_missing_samples(
    frames: list[int], wrist_y: list[float | None]
) -> None:
    pose = complete_pose()
    pose["source_frames"] = frames
    pose["joints"]["L_Wrist"] = [None if y is None else [-0.5, y, 0.0] for y in wrist_y]
    summary = research.summarize_pose(pose)
    assert summary["wrist_vertical_reversals_by_side"]["L"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hip_vertical_range_body_scales", -0.1, "cannot be negative"),
        ("knee_angle_median_deg", 181.0, "between 0 and 180"),
    ],
)
def test_pose_summary_rejects_out_of_domain_features(
    field: str, value: float, message: str
) -> None:
    summary = research.summarize_pose(complete_pose())
    summary[field] = value
    with pytest.raises(research.ResearchError, match=message):
        research.validate_pose_summary(summary)


def test_pose_summary_requires_explicit_nullable_features() -> None:
    summary = research.summarize_pose(complete_pose())
    del summary["knee_angle_median_deg"]
    with pytest.raises(research.ResearchError, match="missing required features"):
        research.validate_pose_summary(summary)


def test_pose_summary_rejects_negative_wrist_reversal_count() -> None:
    summary = research.summarize_pose(complete_pose())
    summary["wrist_vertical_reversals_by_side"]["L"] = -1
    with pytest.raises(research.ResearchError, match="nonnegative integer"):
        research.validate_pose_summary(summary)


def test_preflight_is_offline_and_covers_synthetic_geometry_and_response_schema() -> None:
    research.synthetic_preflight()


def test_frozen_stage_runs_from_local_pose_json_without_network(tmp_path: Path) -> None:
    pose_file = tmp_path / "pose.json"
    pose_file.write_text('{"frame_id":0,"detections":[]}\n', encoding="utf-8")
    output = tmp_path / "stage.json"
    research.build_frozen_stage(pose_file, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["generating_processing_commit"] == "939d1f0ed30d4f47ed8d3f31b53cda94f481b45d"
    assert result["meta"]["shots"] == []
    assert isinstance(result["payload"], str)


@pytest.mark.parametrize("probability", [-0.01, 1.01, math.nan, math.inf])
def test_noul_responses_reject_invalid_probabilities(probability: float) -> None:
    request = research.build_request(research.summarize_pose(pose_for_heights([0, 1, 0])))
    response = research._fixture_response(request)
    response["answers"]["action_deep_bend"]["noul"] = probability
    with pytest.raises(research.ResearchError, match=r"between 0 and 1|finite"):
        research.validate_response(response, request)


def test_choice_response_requires_complete_probability_distribution() -> None:
    request = research.build_request(research.summarize_pose(pose_for_heights([0, 1, 0])))
    response = research._fixture_response(request)
    response["answers"]["action_stage"]["probabilities"].pop("other_or_ambiguous")
    with pytest.raises(research.ResearchError, match="cover every criterion"):
        research.validate_response(response, request)


def test_choice_response_rejects_probabilities_that_do_not_sum_to_one() -> None:
    request = research.build_request(research.summarize_pose(pose_for_heights([0, 1, 0])))
    response = research._fixture_response(request)
    response["answers"]["action_stage"]["probabilities"]["other_or_ambiguous"] = 0.4
    with pytest.raises(research.ResearchError, match="sum to 1"):
        research.validate_response(response, request)


def test_score_response_uses_each_documented_level_and_probability() -> None:
    request = {
        "model": research.MODEL,
        "state": {"value": "fixture"},
        "questions": {
            "severity": {"type": "score", "instructions": "rate it", "criteria": ["low", "high"]}
        },
    }
    response = {
        "model": "typesafe/jev-1.13-fixture",
        "answers": {
            "severity": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.5,
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    research.validate_response(response, request)
    response["answers"]["severity"]["probabilities"] = {"0": 0.1, "1": 0.2}
    with pytest.raises(research.ResearchError, match="sum to 1"):
        research.validate_response(response, request)


def test_invalid_score_request_levels_are_rejected() -> None:
    request = {
        "model": research.MODEL,
        "state": "fixture",
        "questions": {
            "severity": {"type": "score", "instructions": "rate it", "criteria": ["low"]}
        },
    }
    with pytest.raises(research.ResearchError, match="2 to 10 levels"):
        research.validate_request(request)


def test_exact_replay_rows_reconstruct_original_typed_calls_without_search_logic(
    tmp_path: Path,
) -> None:
    question = {
        "generic_unreliable": {
            "type": "noul",
            "instructions": "Is this pose window unreliable?",
        }
    }
    request = {
        "model": research.MODEL,
        "state": {"candidate": "numeric fixture"},
        "questions": question,
    }
    response = {
        "model": "typesafe/jev-1.13-fixture",
        "answers": {"generic_unreliable": {"type": "noul", "noul": 0.2}},
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    for filename in research.RESULT_FILES:
        row = {
            "status": "ok",
            "experiment": filename.removesuffix("_v1.jsonl"),
            "experiment_item_id": "fixture-item",
            "model_requested": request["model"],
            "state": request["state"],
            "questions": request["questions"],
            "request_sha256": research.canonical_hash(request),
            "state_sha256": research.canonical_hash(request["state"]),
            "questions_sha256": research.canonical_hash(request["questions"]),
            "question_count": len(request["questions"]),
            "response": response,
        }
        (tmp_path / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = research._replay_rows(tmp_path)
    assert len(rows) == len(research.RESULT_FILES)
    assert rows[0]["request"] == request
    assert rows[0]["response"] == response
    assert rows[0]["request_sha256"] == research.canonical_hash(request)


def test_summary_keeps_manifest_record_counts_separate_from_choice_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_records = {
        name: (2 if name == "action_quality_v1.jsonl" else 1) for name in research.RESULT_FILES
    }
    monkeypatch.setattr(research, "verify_artifacts", lambda _directory: expected_records)

    def fixture_rows(path: Path) -> list[dict[str, Any]]:
        row_count = expected_records[path.name]
        question = {
            "action_quality_v1.jsonl": ("action_stage", "choice", "same_label", 3),
            "spec_assist_v1.jsonl": ("best_character", "choice", "catalog", 3),
            "quality_generic_v1.jsonl": ("generic_unreliable", "noul", 0.1, 1),
            "injected_corruption_v1.jsonl": ("detector", "noul", 0.9, 6),
        }[path.name]
        cost_by_file = {
            "action_quality_v1.jsonl": 0.001,
            "spec_assist_v1.jsonl": 0.0005,
            "quality_generic_v1.jsonl": 0.0001,
            "injected_corruption_v1.jsonl": 0.0004,
        }
        input_tokens_by_file = {
            "action_quality_v1.jsonl": 4,
            "spec_assist_v1.jsonl": 5,
            "quality_generic_v1.jsonl": 1,
            "injected_corruption_v1.jsonl": 2,
        }
        output_tokens_by_file = {
            "action_quality_v1.jsonl": 3,
            "spec_assist_v1.jsonl": 2,
            "quality_generic_v1.jsonl": 1,
            "injected_corruption_v1.jsonl": 2,
        }
        rows = []
        for _ in range(row_count):
            question_id, answer_type, value, question_count = question
            answer = (
                {"type": answer_type, "choice": value}
                if answer_type == "choice"
                else {"type": answer_type, "noul": value}
            )
            rows.append(
                {
                    "question_count": question_count,
                    "response": {
                        "answers": {question_id: answer},
                        "usage": {
                            "input_tokens": input_tokens_by_file[path.name],
                            "output_tokens": output_tokens_by_file[path.name],
                            "cost": cost_by_file[path.name],
                        },
                    },
                }
            )
        return rows

    monkeypatch.setattr(research, "_jsonl_rows", fixture_rows)
    summary = research.summarize_artifacts(tmp_path)
    assert summary["records"] == expected_records
    assert summary["choice_labels"] == {
        "action_quality_v1.jsonl": {"action_stage": {"same_label": 2}},
        "spec_assist_v1.jsonl": {"best_character": {"catalog": 1}},
    }
    assert summary["usage"] == {
        "requests": 5,
        "judgments": 16,
        "input_tokens": 16,
        "output_tokens": 11,
        "provider_cost_usd": 0.003,
    }


def test_malformed_result_schema_fails_before_replay_output_changes(tmp_path: Path) -> None:
    (tmp_path / research.RESULT_FILES[0]).write_text('{"status":"ok"}\n', encoding="utf-8")
    output = tmp_path / "replay.jsonl"
    output.write_text("keep-existing\n", encoding="utf-8")
    with pytest.raises(research.ResearchError, match=r"request\.model must be"):
        research._replay_rows(tmp_path)
    assert output.read_text(encoding="utf-8") == "keep-existing\n"


def test_artifact_manifest_detects_byte_changes_before_replay_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "artifacts"
    directory.mkdir()
    source_root = Path(research.__file__).parent
    processing_sources = (
        "frozen_stage.py",
        "frozen_nova77.py",
        "frozen_processing.py",
        "nova77.py",
    )
    for name in processing_sources:
        shutil.copyfile(source_root / name, tmp_path / name)
    artifacts: dict[str, bytes] = {
        "windows.json": b'{"schema_version":"fixture"}\n',
        "source.json": b'{"schema_version":"fixture"}\n',
    }
    request = {
        "model": research.MODEL,
        "state": {"candidate": {"digest": {"fixture": 1}}},
        "questions": {"quality": {"type": "noul", "instructions": "Is it valid?"}},
    }
    result = {
        "status": "ok",
        "experiment": "fixture",
        "model_requested": request["model"],
        "state": request["state"],
        "questions": request["questions"],
        "request_sha256": research.canonical_hash(request),
        "state_sha256": research.canonical_hash(request["state"]),
        "questions_sha256": research.canonical_hash(request["questions"]),
        "question_count": 1,
        "response": {
            "model": "typesafe/jev-1.13-fixture",
            "answers": {"quality": {"type": "noul", "noul": 1.0}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    for filename in research.RESULT_FILES:
        artifacts[filename] = (json.dumps(result) + "\n").encode()
    for filename, payload in artifacts.items():
        (directory / filename).write_bytes(payload)
    manifest_path = tmp_path / "manifest.toml"
    manifest_lines = ["schema_version = 1", 'generating_processing_commit = "fixture"', ""]
    for name in processing_sources:
        digest = research._sha256_bytes((tmp_path / name).read_bytes())
        manifest_lines.extend([f'[processing_sources."{name}"]', f'sha256 = "{digest}"', ""])
    for name, payload in artifacts.items():
        manifest_lines.extend(
            [f'[artifacts."{name}"]', f'sha256 = "{research._sha256_bytes(payload)}"', ""]
        )
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    monkeypatch.setattr(research, "MANIFEST", manifest_path)
    assert research.verify_artifacts(directory) == {
        "windows.json": 1,
        "source.json": 1,
        **dict.fromkeys(research.RESULT_FILES, 1),
    }

    output = tmp_path / "replay.jsonl"
    output.write_text("preserve-existing\n", encoding="utf-8")
    (directory / "windows.json").write_bytes(b'{"schema_version":"altered"}\n')
    with pytest.raises(research.ResearchError, match=r"checksum mismatch: windows\.json"):
        research.replay_artifacts(directory, output)
    assert output.read_text(encoding="utf-8") == "preserve-existing\n"

    (directory / "windows.json").write_bytes(artifacts["windows.json"])
    (tmp_path / "nova77.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(research.ResearchError, match="frozen processing source checksum mismatch"):
        research.verify_artifacts(directory)


def test_search_and_arbitrary_query_flags_are_not_cli_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as search_error:
        research.main(["search"])
    assert search_error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    with pytest.raises(SystemExit) as query_error:
        research.main(
            [
                "classify",
                "--pose",
                "pose.json",
                "--cache",
                "cache.jsonl",
                "--output",
                "out.json",
                "--query",
                "anything",
            ]
        )
    assert query_error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_exact_cached_request_replays_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pose = pose_for_heights([0.0, 1.0, 0.0])
    pose_summary = research.summarize_pose(pose)
    request = research.build_request(pose_summary)
    cache_row = {
        "request_sha256": research.canonical_hash(request),
        "request": request,
        "response": research._fixture_response(request),
    }
    cache = tmp_path / "responses.jsonl"
    cache.write_text(json.dumps(cache_row) + "\n", encoding="utf-8")
    output = tmp_path / "result.json"
    result = research.classify(pose, cache, output)
    assert result["cached"] is True
    assert (
        json.loads(output.read_text(encoding="utf-8"))["request_sha256"]
        == cache_row["request_sha256"]
    )


def test_invalid_cache_fails_before_network_or_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "responses.jsonl"
    cache.write_text('{"request": {}}\n', encoding="utf-8")
    pose = pose_for_heights([0.0, 1.0, 0.0])
    pose_summary = research.summarize_pose(pose)
    request = research.build_request(pose_summary)
    output = tmp_path / "existing.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": "pose-classification-result-1",
                "request_sha256": research.canonical_hash(request),
                "summary": pose_summary,
                "response": research._fixture_response(request),
            }
        ),
        encoding="utf-8",
    )
    previous_output = output.read_text(encoding="utf-8")

    def no_network(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("network was called after invalid cache input")

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(research.urllib.request, "urlopen", no_network)
    with pytest.raises(research.ResearchError, match="request must contain exactly"):
        research.classify(pose, cache, output)
    assert cache.read_text(encoding="utf-8") == '{"request": {}}\n'
    assert output.read_text(encoding="utf-8") == previous_output


def test_malformed_existing_summary_fails_before_network_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pose = pose_for_heights([0.0, 1.0, 0.0])
    summary = research.summarize_pose(pose)
    request = research.build_request(summary)
    cache = tmp_path / "responses.jsonl"
    output = tmp_path / "existing.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": "pose-classification-result-1",
                "request_sha256": research.canonical_hash(request),
                "summary": {"schema_version": "broken"},
                "response": research._fixture_response(request),
            }
        ),
        encoding="utf-8",
    )
    previous_output = output.read_text(encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(
        research.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network called after malformed output"),
    )
    with pytest.raises(research.ResearchError, match="pose-summary-1"):
        research.classify(pose, cache, output)
    assert not cache.exists()
    assert output.read_text(encoding="utf-8") == previous_output


def test_invalid_provider_response_writes_neither_cache_nor_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadResponse:
        def __enter__(self) -> BadResponse:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"model":"fixture","answers":{},"usage":{}}'

    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.setattr(research.urllib.request, "urlopen", lambda *_args, **_kwargs: BadResponse())
    cache = tmp_path / "responses.jsonl"
    output = tmp_path / "result.json"
    with pytest.raises(research.ResearchError, match="exactly the requested question ids"):
        research.classify(pose_for_heights([0.0, 1.0, 0.0]), cache, output)
    assert not cache.exists()
    assert not output.exists()


def test_nonfinite_pose_fails_before_outputs_are_touched(tmp_path: Path) -> None:
    pose = pose_for_heights([0.0, math.nan, 0.0])
    cache = tmp_path / "cache.jsonl"
    output = tmp_path / "output.json"
    with pytest.raises(research.ResearchError, match="finite"):
        research.classify(pose, cache, output)
    assert not cache.exists()
    assert not output.exists()
