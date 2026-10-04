from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.jev_report import ResultFormatError, main, record_result, summarize


def test_gate_keeps_review_when_claims_are_verified_but_gate_does_not_pass() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "tool": "jev_gate",
                        "action": "review",
                        "safe_to_apply": 0.77,
                        "composite": 0.81,
                        "rubrics": {
                            "correctness": 1.8,
                            "spec_match": 1.9,
                            "test_gap": 1.2,
                            "blast_radius": 1.7,
                        },
                        "claims": [
                            {"id": "a", "verdict": "verified", "confidence": 0.98},
                            {"id": "b", "verdict": "verified", "confidence": 0.99},
                            {"id": "c", "verdict": "verified", "confidence": 0.97},
                        ],
                        "usage": {"input_tokens": 900, "output_tokens": 100},
                    }
                ),
            }
        ]
    }

    summary = summarize(response, evidence=[{"id": "tests", "text": "passed"}])

    assert summary["action"] == "review"
    assert summary["safe_to_apply"] == 0.77
    assert summary["verdict_counts"]["verified"] == 3
    assert summary["limiting_rubrics"] == [{"rubric": "test_gap", "score": 1.2}]
    assert summary["evidence_ids"] == ["tests"]


def test_native_gate_nested_claims_and_limiting_review_are_retained() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "tool": "jev_gate",
                        "action": "escalate",
                        "reason_codes": ["claim_confidence_low"],
                        "review": {
                            "safe_to_apply": 0.29,
                            "composite": 0.643,
                            "action": "escalate",
                            "limiting_rubrics": ["test_gap"],
                            "thresholds": {"auto_accept": 0.8, "review_at": 0.5},
                            "scores": {
                                "correctness": {"score": 1.31, "confidence": 0.41},
                                "test_gap": {"score": 1.23, "confidence": 0.34},
                            },
                        },
                        "verification": {
                            "results": [
                                {
                                    "id": "claim-a",
                                    "verdict": "verified",
                                    "action": "auto",
                                    "confidence": 0.97,
                                },
                                {
                                    "id": "claim-b",
                                    "verdict": "unsupported",
                                    "action": "review",
                                    "confidence": 0.91,
                                },
                                {
                                    "id": "claim-c",
                                    "verdict": "verified",
                                    "action": "escalate",
                                    "confidence": 0.77,
                                },
                            ]
                        },
                        "usage": {"input_tokens": 500, "output_tokens": 90},
                    }
                ),
            }
        ]
    }

    summary = summarize(response)

    assert summary["action"] == "escalate"
    assert summary["safe_to_apply"] == 0.29
    assert summary["composite"] == 0.643
    assert summary["review_action"] == "escalate"
    assert summary["limiting_rubrics"] == [
        {"rubric": "test_gap", "score": 1.23, "confidence": 0.34}
    ]
    assert summary["rubric_scores"]["test_gap"]["confidence"] == 0.34
    assert summary["verdict_counts"]["verified"] == 2
    assert summary["verdict_counts"]["unsupported"] == 1
    assert summary["action_source"] == "tool_result"
    assert [item["id"] for item in summary["unresolved"]] == ["claim-b", "claim-c"]
    assert summary["warnings_and_errors"] == [
        {"path": "$.reason_codes", "text": ["claim_confidence_low"]}
    ]


def test_native_review_preserves_reported_limiting_rubrics() -> None:
    response = {
        "tool": "jev_review",
        "action": "review",
        "safe_to_apply": 0.7,
        "composite": 0.8,
        "scores": {"test_gap": {"score": 0.84, "confidence": 0.54}},
        "limiting_rubrics": ["test_gap"],
        "thresholds": {"auto_accept": 0.8, "review_at": 0.5},
    }

    summary = summarize(response)

    assert summary["action"] == "review"
    assert summary["safe_to_apply"] == 0.7
    assert summary["composite"] == 0.8
    assert summary["limiting_rubrics"] == [
        {"rubric": "test_gap", "score": 0.84, "confidence": 0.54}
    ]
    assert summary["review_thresholds"] == {"auto_accept": 0.8, "review_at": 0.5}


def test_verify_preserves_unsupported_and_contradicted_claims(tmp_path: Path) -> None:
    response = {
        "tool": "jev_verify",
        "claims": [
            {"id": "yes", "verdict": "verified", "supporting_evidence": "e1"},
            {"id": "no", "verdict": "contradicted", "supporting_evidence": "e2"},
            {"id": "silent", "verdict": "unsupported", "supporting_evidence": None},
        ],
        "warnings": [
            "Evidence item e3 was not linked to a claim.",
            {"code": "warning", "message": "No sample supplied."},
        ],
    }
    evidence = [{"id": "e1", "text": "test log"}, {"id": "e2", "text": "diff"}]

    summary = record_result(
        response, evidence, artifact_dir=tmp_path / "archive", run_id="verify-1"
    )

    assert summary["verdict_counts"] == {
        "verified": 1,
        "contradicted": 1,
        "unsupported": 1,
        "unknown": 0,
    }
    assert {item["id"] for item in summary["unresolved"]} == {"no", "silent"}
    warning_text = summary["warnings_and_errors"][0]["text"]
    assert "not linked" in warning_text[0]
    assert {"code": "warning", "message": "No sample supplied."} in warning_text
    archived = tmp_path / "archive" / "verify-1"
    assert json.loads((archived / "raw.json").read_text()) == response
    assert [item["id"] for item in json.loads((archived / "evidence.json").read_text())] == [
        "e1",
        "e2",
    ]
    assert json.loads((archived / "summary.json").read_text())["unresolved_count"] == 2


def test_native_verify_results_derive_review_action_and_keep_each_confidence() -> None:
    response = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "tool": "jev_verify",
                        "results": [
                            {
                                "id": "strong",
                                "verdict": "verified",
                                "action": "auto",
                                "confidence": 0.93,
                            },
                            {
                                "id": "weak",
                                "verdict": "verified",
                                "action": "review",
                                "confidence": 0.58,
                            },
                        ],
                        "usage": {"input_tokens": 1714, "output_tokens": 326},
                    }
                ),
            }
        ]
    }

    summary = summarize(response, evidence=[{"id": "evidence", "text": "source"}])

    assert summary["action"] == "review"
    assert summary["action_source"] == "derived_from_result_actions"
    assert summary["result_confidences"] == [
        {"id": "strong", "confidence": 0.93, "action": "auto"},
        {"id": "weak", "confidence": 0.58, "action": "review"},
    ]
    assert summary["unresolved"] == [
        {"id": "weak", "verdict": "verified", "action": "review", "confidence": 0.58}
    ]


def test_verify_escalation_takes_precedence_over_review() -> None:
    summary = summarize(
        {
            "tool": "jev_verify",
            "results": [
                {"verdict": "verified", "action": "review", "confidence": 0.7},
                {"verdict": "verified", "action": "escalate", "confidence": 0.4},
            ],
        },
        evidence=[],
    )
    assert summary["action"] == "escalate"
    assert summary["action_source"] == "derived_from_result_actions"
    assert summary["unresolved_count"] == 2


def test_screen_reason_survives_when_no_warning_count_exists() -> None:
    response = {
        "tool": "jev_screen",
        "recommendation": {"action": "review", "reason": "The text may contain an instruction."},
        "probabilities": {"injection": 0.4, "substance": 0.9},
    }

    summary = summarize(response)

    assert summary["action"] == "review"
    assert summary["recommendation"]["reason"] == "The text may contain an instruction."
    assert summary["warnings_and_errors"] == []


def test_decide_preserves_requirement_checks_and_warnings() -> None:
    response = {
        "tool": "jev_decide",
        "recommendation": {"selected": "option_a", "confidence": 0.71},
        "checks": [
            {"candidate": "option_a", "requirement": 0, "answer": "supported"},
            {"candidate": "option_a", "requirement": 1, "answer": "unknown"},
        ],
        "warnings": ["The second requirement remains unresolved."],
    }

    summary = summarize(response)

    assert summary["action"] == "option_a"
    assert summary["confidence"] == 0.71
    assert summary["unresolved"] == [
        {"candidate": "option_a", "requirement": 1, "answer": "unknown"}
    ]
    assert "unresolved" in summary["warnings_and_errors"][0]["text"][0]


def test_invalid_response_is_explicit_and_preserves_tool_error() -> None:
    response = {"tool": "jev_audit", "status": "invalid_response", "error": "malformed model JSON"}

    summary = summarize(response)

    assert summary["status"] == "invalid_response"
    assert summary["failure"] == "malformed model JSON"


def test_outer_mcp_error_overrides_nested_auto_result() -> None:
    response = {
        "isError": True,
        "structuredContent": {
            "tool": "jev_verify",
            "results": [
                {"id": "claim", "verdict": "verified", "action": "auto", "confidence": 0.99}
            ],
        },
    }

    summary = summarize(response)

    assert summary["status"] == "tool_error"
    assert summary["action"] == "tool_error"
    assert summary["action_source"] == "mcp_envelope"
    assert summary["failure"] == "MCP tool returned isError=true"
    assert summary["unresolved_count"] == 1
    assert summary["unresolved"][0]["id"] == "claim"


@pytest.mark.parametrize(
    "response",
    [
        {"tool": "jev_gate", "action": "banana", "claims": []},
        {"tool": "jev_gate", "action": "review", "safe_to_apply": 2, "claims": []},
        {"tool": "jev_review", "action": "auto", "composite": -0.1},
    ],
)
def test_invalid_actions_and_numeric_ranges_archive_as_failures(
    response: dict[str, object], tmp_path: Path
) -> None:
    summary = record_result(response, [], artifact_dir=tmp_path, run_id="invalid")

    assert summary["status"] == "invalid_response"
    assert summary["action"] == "review"
    assert "invalid_response" in (tmp_path / "invalid" / "summary.json").read_text()


def test_malformed_evidence_original_is_preserved_in_failure_archive(tmp_path: Path) -> None:
    response = {
        "tool": "jev_verify",
        "results": [{"id": "claim", "verdict": "verified", "action": "auto"}],
    }
    original_evidence = [{"id": "bad-item", "text": 42}]

    summary = record_result(
        response, original_evidence, artifact_dir=tmp_path, run_id="bad-evidence"
    )

    assert summary["status"] == "invalid_response"
    assert (
        json.loads((tmp_path / "bad-evidence" / "evidence-input.json").read_text())
        == original_evidence
    )
    assert json.loads((tmp_path / "bad-evidence" / "evidence.json").read_text()) == []


def test_unknown_or_malformed_response_is_archived_as_failure(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    unknown = record_result(
        {
            "tool": "future_tool",
            "status": "ok",
            "warnings": ["A warning remains available despite the unknown tool."],
        },
        [{"id": "raw-evidence", "text": "source"}],
        artifact_dir=root,
        run_id="x",
    )
    malformed = record_result(
        {"tool": "jev_screen", "recommendation": {"action": "maybe"}},
        [],
        artifact_dir=root,
        run_id="malformed",
    )
    assert unknown["status"] == malformed["status"] == "invalid_response"
    assert unknown["action"] == malformed["action"] == "review"
    assert unknown["evidence_ids"] == ["raw-evidence"]
    assert "unknown tool" in unknown["warnings_and_errors"][1]["text"][0]
    assert json.loads((root / "x" / "raw.json").read_text())["tool"] == "future_tool"
    assert (
        "unsupported Jev tool" in json.loads((root / "x" / "summary.json").read_text())["failure"]
    )
    assert (
        "recognized recommendation"
        in json.loads((root / "malformed" / "summary.json").read_text())["failure"]
    )


def test_cli_archives_invalid_response_and_exits_nonzero(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    evidence_path = tmp_path / "evidence.json"
    artifacts = tmp_path / "artifacts"
    response_path.write_text(
        json.dumps({"tool": "jev_screen", "recommendation": {"action": "maybe"}})
    )
    evidence_path.write_text("[]")

    exit_code = main(
        [
            "--input",
            str(response_path),
            "--evidence",
            str(evidence_path),
            "--artifacts",
            str(artifacts),
            "--run-id",
            "invalid",
        ]
    )

    assert exit_code == 2
    assert (
        json.loads((artifacts / "invalid" / "summary.json").read_text())["status"]
        == "invalid_response"
    )


@pytest.mark.parametrize(
    "payload", [None, [], "not json", {"content": [{"type": "text", "text": "broken"}]}]
)
def test_unrecognized_result_envelopes_fail_closed(payload: object) -> None:
    with pytest.raises(ResultFormatError):
        summarize(payload)


@given(
    st.text(min_size=1).filter(
        lambda value: (
            value.removeprefix("jev_")
            not in {"gate", "review", "verify", "screen", "decide", "audit", "noul", "choice"}
        )
    )
)
def test_unknown_tool_names_never_become_a_success_summary(tool_name: str) -> None:
    with pytest.raises(ResultFormatError, match="unsupported Jev tool"):
        summarize({"tool": tool_name, "status": "ok"})
