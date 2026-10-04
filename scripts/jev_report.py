#!/usr/bin/env python3
"""Archive Jev tool responses and print a compact, fail-closed summary.

This module deliberately does not call Jev or decide whether a response is good
enough. It records what the tool returned and keeps review states visible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPPORTED_TOOLS = {"gate", "review", "verify", "screen", "decide", "audit", "noul", "choice"}
VERDICTS = {"verified", "contradicted", "unsupported", "unknown"}
_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")


class ResultFormatError(ValueError):
    """Raised when a tool response cannot be recognized without guessing."""


def _jsonable(value: Any) -> Any:
    """Convert common MCP result objects to JSON-compatible values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    raise ResultFormatError(f"unsupported response object type: {type(value).__name__}")


def _unwrap(raw: Any) -> dict[str, Any]:
    """Unwrap an MCP CallToolResult, including a JSON text content block."""
    result = _jsonable(raw)
    if not isinstance(result, dict):
        raise ResultFormatError("tool response must be a JSON object or MCP result object")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        payload = dict(structured)
        return _carry_mcp_error(result, payload)
    content = result.get("content")
    if isinstance(content, list):
        text_blocks = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if len(text_blocks) == 1 and isinstance(text_blocks[0], str):
            try:
                decoded = json.loads(text_blocks[0])
            except json.JSONDecodeError as exc:
                raise ResultFormatError("MCP text content is not valid JSON") from exc
            if isinstance(decoded, dict):
                return _carry_mcp_error(result, decoded)
        # Some Jev tools return content blocks without an explicit MCP wrapper.
    if isinstance(result.get("tool"), str):
        return result
    raise ResultFormatError("MCP result has no recognized structuredContent or JSON text block")


def _carry_mcp_error(envelope: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Retain transport failures when unwrapping native MCP content."""
    if envelope.get("isError") is True:
        payload["_mcp_is_error"] = True
        payload["_mcp_error"] = envelope.get("error", "MCP tool returned isError=true")
    for key in ("warnings", "errors", "error"):
        if key in envelope and key not in payload:
            payload[f"_mcp_{key}"] = envelope[key]
    return payload


def _tool_name(value: str | None, payload: dict[str, Any]) -> str:
    name = value or payload.get("tool")
    if not isinstance(name, str):
        raise ResultFormatError(
            "tool name is required (gate, review, verify, screen, decide, audit, noul, choice)"
        )
    # MCP envelopes may identify the tool as jev_gate or mcp__jev__jev_gate.
    normalized = name.rsplit("__", 1)[-1].removeprefix("jev_")
    if normalized not in SUPPORTED_TOOLS:
        raise ResultFormatError(f"unsupported Jev tool name: {name}")
    return normalized


def _validate_probabilities(
    payload: dict[str, Any], name: str, records: list[dict[str, Any]]
) -> None:
    """Reject malformed values whose meaning the compact summary exposes."""
    for key in ("confidence", "safe_to_apply"):
        value = payload.get(key)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1
        ):
            raise ResultFormatError(f"{key} must be a number from 0 through 1")
    review = payload.get("review")
    if isinstance(review, dict):
        safe = review.get("safe_to_apply")
        if safe is not None and (
            not isinstance(safe, (int, float)) or isinstance(safe, bool) or not 0 <= safe <= 1
        ):
            raise ResultFormatError("review.safe_to_apply must be a number from 0 through 1")
        composite = review.get("composite")
        if composite is not None and (
            not isinstance(composite, (int, float))
            or isinstance(composite, bool)
            or not 0 <= composite <= 2
        ):
            raise ResultFormatError("review.composite must be a number from 0 through 2")
        review_action = review.get("action")
        if review_action is not None and review_action not in {"auto", "review", "escalate"}:
            raise ResultFormatError(f"unrecognized nested review action: {review_action}")
    composite = payload.get("composite")
    if composite is not None and (
        not isinstance(composite, (int, float))
        or isinstance(composite, bool)
        or not 0 <= composite <= 2
    ):
        raise ResultFormatError("composite must be a number from 0 through 2")
    allowed_actions = {
        "gate": {"auto", "review", "escalate"},
        "review": {"auto", "review", "escalate"},
        "verify": {"auto", "review", "escalate"},
        "screen": {"pass", "review", "block", "skip"},
        "audit": {"pass", "review", "escalate"},
    }
    action = payload.get("action")
    if action is not None and name in allowed_actions and action not in allowed_actions[name]:
        raise ResultFormatError(f"unrecognized {name} action: {action}")
    review = payload.get("review")
    if isinstance(review, dict) and isinstance(review.get("scores"), dict):
        for rubric, values in review["scores"].items():
            if isinstance(values, dict):
                score = values.get("score")
                confidence = values.get("confidence")
                if score is not None and (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not 0 <= score <= 2
                ):
                    raise ResultFormatError(f"review score for {rubric} must be from 0 through 2")
                if confidence is not None and (
                    not isinstance(confidence, (int, float))
                    or isinstance(confidence, bool)
                    or not 0 <= confidence <= 1
                ):
                    raise ResultFormatError(
                        f"review confidence for {rubric} must be from 0 through 1"
                    )
    top_confidence = payload.get("confidence")
    if top_confidence is None and isinstance(payload.get("recommendation"), dict):
        top_confidence = payload["recommendation"].get("confidence")
    if top_confidence is not None and (
        not isinstance(top_confidence, (int, float))
        or isinstance(top_confidence, bool)
        or not 0 <= top_confidence <= 1
    ):
        raise ResultFormatError("recommendation confidence must be a number from 0 through 1")
    if name in {"decide", "choice"} and isinstance(payload.get("recommendation"), dict):
        selected = payload["recommendation"].get("selected")
        if selected is not None and not isinstance(selected, str):
            raise ResultFormatError("recommendation.selected must be a string or null")
    allowed_result_actions = (
        {"ok", "wrong", "invalid_response"}
        if name == "audit"
        else {"auto", "review", "escalate", "wrong", "invalid_response"}
    )
    for item in records:
        action = item.get("action")
        if action is not None and action not in allowed_result_actions:
            raise ResultFormatError(f"unrecognized per-result action: {action}")
        confidence = item.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise ResultFormatError("per-result confidence must be a number from 0 through 1")


def _get_action(payload: dict[str, Any], tool: str) -> str | None:
    if tool in {"gate", "review"}:
        value = payload.get("action")
    elif tool == "screen":
        recommendation = payload.get("recommendation")
        value = recommendation.get("action") if isinstance(recommendation, dict) else None
    elif tool == "audit":
        value = payload.get("action")
    elif tool in {"decide", "choice"}:
        recommendation = payload.get("recommendation")
        value = recommendation.get("selected") if isinstance(recommendation, dict) else None
    elif tool in {"verify", "noul"}:
        value = None
    else:
        value = None
    return value if isinstance(value, str) else None


def _gate_claims(payload: dict[str, Any]) -> list[Any] | None:
    for key in ("claims", "claim_results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    verification = payload.get("verification")
    if isinstance(verification, dict) and isinstance(verification.get("results"), list):
        return verification["results"]
    return None


def _records(payload: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    keys = {
        "gate": ("claims", "claim_results", "results"),
        "verify": ("claims", "results"),
        "noul": ("results", "propositions"),
        "audit": ("records", "checks"),
        "decide": ("checks",),
        "choice": ("checks",),
        "screen": ("warnings", "errors"),
        "review": ("warnings", "errors"),
    }[tool]
    if tool == "gate":
        value = _gate_claims(payload)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _claim_verdict(item: dict[str, Any]) -> str | None:
    value = item.get("verdict")
    if isinstance(value, str) and value in VERDICTS:
        return value
    return None


def _count_verdicts(payload: dict[str, Any], tool: str) -> dict[str, int]:
    counts = dict.fromkeys(("verified", "contradicted", "unsupported", "unknown"), 0)
    if payload.get("status") == "invalid_response":
        return counts
    if tool == "gate":
        raw_claims = _gate_claims(payload)
        if raw_claims is None:
            raise ResultFormatError("gate response is missing its claims list")
        for item in raw_claims:
            if not isinstance(item, dict):
                raise ResultFormatError("gate claims must be objects")
            verdict = _claim_verdict(item)
            if verdict is None:
                raise ResultFormatError("gate claim has no recognized verdict")
            counts[verdict] += 1
    elif tool == "verify":
        raw_claims = payload.get("claims", payload.get("results"))
        if not isinstance(raw_claims, list):
            raise ResultFormatError("verify response is missing its claims list")
        for item in raw_claims:
            if not isinstance(item, dict):
                raise ResultFormatError("verify claims must be objects")
            verdict = _claim_verdict(item)
            if verdict is None:
                raise ResultFormatError("verify claim has no recognized verdict")
            counts[verdict] += 1
    return counts


def _collect_text_fields(payload: Any, keys: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def visit(node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}"
                if key in keys and value not in (None, "", [], {}):
                    found.append({"path": child, "text": value})
                else:
                    visit(value, child)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]")

    visit(payload)
    # Deduplicate the repeated warnings list visible through parent traversal.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in found:
        unique[(entry["path"], json.dumps(entry["text"], sort_keys=True, ensure_ascii=False))] = (
            entry
        )
    return list(unique.values())


def _evidence_items(evidence: Any) -> list[dict[str, str]]:
    evidence = _jsonable(evidence)
    if isinstance(evidence, str):
        return [{"id": "evidence-1", "text": evidence}]
    if isinstance(evidence, dict):
        if isinstance(evidence.get("text"), str):
            return [{"id": str(evidence.get("id", "evidence-1")), "text": evidence["text"]}]
        return [{"id": str(key), "text": str(value)} for key, value in evidence.items()]
    if isinstance(evidence, list):
        result: list[dict[str, str]] = []
        for index, item in enumerate(evidence, start=1):
            if isinstance(item, str):
                result.append({"id": f"evidence-{index}", "text": item})
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                result.append(
                    {"id": str(item.get("id", f"evidence-{index}")), "text": item["text"]}
                )
            else:
                raise ResultFormatError(f"evidence item {index} must be labeled text or a string")
        return result
    raise ResultFormatError("evidence must be text, a labeled item, a mapping, or a list")


def summarize(raw: Any, *, tool: str | None = None, evidence: Any = None) -> dict[str, Any]:
    """Return a truthful compact summary; unknown and invalid shapes raise."""
    payload = _unwrap(raw)
    name = _tool_name(tool, payload)
    if payload.get("status") == "invalid_response":
        status = "invalid_response"
    elif payload.get("isError") is True or payload.get("_mcp_is_error") is True:
        status = "tool_error"
    else:
        status = str(payload.get("status", "ok"))
    action = _get_action(payload, name)
    records = _records(payload, name)
    if status == "tool_error":
        action = "tool_error"
        action_source = "mcp_envelope"
    else:
        action_source = "tool_result" if action is not None else None
        _validate_probabilities(payload, name, records)
    if status != "invalid_response" and action is None and name in {"verify", "gate"} and records:
        result_actions = [item.get("action") for item in records]
        if "escalate" in result_actions:
            action = "escalate"
            action_source = "derived_from_result_actions"
        elif any(value in {"review", "invalid_response"} for value in result_actions):
            action = "review"
            action_source = "derived_from_result_actions"
        elif all(value == "auto" for value in result_actions):
            action = "auto"
            action_source = "derived_from_result_actions"
    verdicts = _count_verdicts(payload, name)
    rubrics = payload.get("rubrics", payload.get("scores"))
    limiting: list[dict[str, Any]] = []
    if status == "invalid_response":
        pass
    elif name in {"gate", "review"}:
        if not isinstance(action, str):
            raise ResultFormatError(f"{name} response is missing an action")
        score_source = (
            rubrics
            if isinstance(rubrics, dict)
            else {
                key: payload[key]
                for key in ("correctness", "spec_match", "test_gap", "blast_radius")
                if key in payload
            }
        )
        numeric_scores = (
            [
                (key, val)
                for key, val in score_source.items()
                if isinstance(val, (int, float)) and not isinstance(val, bool)
            ]
            if isinstance(score_source, dict)
            else []
        )
        if numeric_scores:
            min_score = min(value for _, value in numeric_scores)
            limiting = [
                {"rubric": key, "score": value}
                for key, value in numeric_scores
                if value == min_score
            ]
    elif name == "screen":
        recommendation = payload.get("recommendation")
        if not isinstance(recommendation, dict) or action not in {
            "pass",
            "review",
            "block",
            "skip",
        }:
            raise ResultFormatError("screen response is missing a recognized recommendation action")
    elif name == "audit":
        if action not in {"pass", "review", "escalate"}:
            raise ResultFormatError("audit response is missing a recognized audit action")
    elif name in {"decide", "choice"}:
        recommendation = payload.get("recommendation")
        if not isinstance(recommendation, dict):
            raise ResultFormatError("decide response is missing recommendation")
    elif name == "noul":
        if not isinstance(payload.get("results"), list):
            raise ResultFormatError("noul response is missing results list")
    elif name == "verify":
        pass

    warnings = _collect_text_fields(
        payload,
        {
            "warnings",
            "errors",
            "error",
            "warning",
            "reason_codes",
            "_mcp_warnings",
            "_mcp_errors",
            "_mcp_error",
        },
    )
    unresolved: list[dict[str, Any]] = []
    for item in records:
        verdict = _claim_verdict(item)
        if (
            verdict in {"contradicted", "unsupported", "unknown"}
            or item.get("action") in {"review", "escalate", "invalid_response", "wrong"}
            or item.get("decision") == "review"
            or item.get("answer") in {"unknown", "contradicted"}
            or item.get("label") == "uncertain"
            or item.get("auto") is False
        ):
            unresolved.append(item)
    if status == "tool_error":
        unresolved = records.copy()
    review_value = payload.get("review")
    review: dict[str, Any] = review_value if isinstance(review_value, dict) else {}
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {key: payload[key] for key in (*_TOKEN_KEYS, "cost") if key in payload}
    normalized_evidence = _evidence_items([] if evidence is None else evidence)
    score_container = review.get("scores") if name == "gate" else None
    if not isinstance(score_container, dict):
        direct_scores = payload.get("scores", payload.get("rubrics"))
        score_container = direct_scores if isinstance(direct_scores, dict) else {}
    reported_limiting = review.get("limiting_rubrics", payload.get("limiting_rubrics"))
    if not limiting and name in {"gate", "review"} and isinstance(reported_limiting, list):
        for rubric in reported_limiting:
            score = score_container.get(rubric)
            entry: dict[str, Any] = {"rubric": rubric}
            if isinstance(score, dict):
                entry.update({key: score[key] for key in ("score", "confidence") if key in score})
            else:
                entry["score"] = score
            limiting.append(entry)
    safe_to_apply = payload.get("safe_to_apply", review.get("safe_to_apply"))
    composite = payload.get("composite", review.get("composite"))
    summary: dict[str, Any] = {
        "tool": name,
        "status": status,
        "action": action,
        "action_source": action_source,
        "verdict_counts": verdicts,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "warnings_and_errors": warnings,
        "limiting_rubrics": limiting,
        "rubric_scores": score_container,
        "review_action": review.get("action", payload.get("action") if name == "review" else None),
        "review_thresholds": review.get("thresholds", payload.get("thresholds")),
        "confidence": payload.get(
            "confidence",
            (payload.get("recommendation") or {}).get("confidence")
            if isinstance(payload.get("recommendation"), dict)
            else None,
        ),
        "result_confidences": [
            {key: item[key] for key in ("id", "confidence", "action") if key in item}
            for item in records
            if "confidence" in item or "action" in item
        ],
        "safe_to_apply": safe_to_apply,
        "composite": composite,
        "recommendation": payload.get("recommendation"),
        "usage": usage,
        "evidence_ids": [item["id"] for item in normalized_evidence],
    }
    if status == "invalid_response":
        summary["failure"] = payload.get(
            "error", payload.get("message", "Jev returned invalid_response")
        )
    if status == "tool_error":
        summary["failure"] = payload.get(
            "_mcp_error", payload.get("error", "MCP tool returned isError=true")
        )
    return summary


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def record_result(
    raw: Any,
    evidence: Any,
    *,
    tool: str | None = None,
    artifact_dir: str | Path = "work/jev-evidence",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Atomically archive a full Jev result and labeled evidence, then return summary."""
    try:
        original_evidence = _jsonable(evidence)
    except ResultFormatError as exc:
        original_evidence = {
            "serialization_error": str(exc),
            "type": type(evidence).__name__,
            "repr": repr(evidence),
        }
    try:
        payload = _unwrap(raw)
        summary = summarize(raw, tool=tool, evidence=evidence)
        normalized_evidence = _evidence_items(evidence)
    except ResultFormatError as exc:
        # Archive malformed results as failures so the raw response and its
        # reported protocol error survive for follow-up inspection.
        payload = {"status": "invalid_response", "error": str(exc)}
        try:
            raw_for_warnings = _jsonable(raw)
            existing_warnings = _collect_text_fields(
                raw_for_warnings,
                {"warnings", "errors", "error", "warning", "reason_codes"},
            )
        except ResultFormatError:
            existing_warnings = []
        summary = {
            "tool": tool,
            "status": "invalid_response",
            "action": "review",
            "action_source": "invalid_response",
            "verdict_counts": dict.fromkeys(
                ("verified", "contradicted", "unsupported", "unknown"), 0
            ),
            "unresolved_count": 0,
            "unresolved": [],
            "warnings_and_errors": [
                {"path": "$.error", "text": str(exc)},
                *existing_warnings,
            ],
            "limiting_rubrics": [],
            "rubric_scores": {},
            "review_action": None,
            "review_thresholds": None,
            "confidence": None,
            "result_confidences": [],
            "safe_to_apply": None,
            "composite": None,
            "usage": {},
            "evidence_ids": [],
            "failure": str(exc),
        }
        try:
            normalized_evidence = _evidence_items(evidence)
        except ResultFormatError:
            normalized_evidence = []
        summary["evidence_ids"] = [item["id"] for item in normalized_evidence]
    archived_raw = _jsonable(raw)
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", identifier):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and hyphen")
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / identifier
    if final_dir.exists():
        raise FileExistsError(f"evidence run already exists: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}.", dir=root))
    try:
        files = {
            "raw.json": _json_bytes(archived_raw),
            "payload.json": _json_bytes(payload),
            "evidence.json": _json_bytes(normalized_evidence),
            "evidence-input.json": _json_bytes(original_evidence),
            "summary.json": _json_bytes(summary),
        }
        for filename, content in files.items():
            (temporary / filename).write_bytes(content)
        os.replace(temporary, final_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary["artifact_path"] = str(final_dir)
    return summary


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON file with the complete Jev/MCP response"
    )
    parser.add_argument(
        "--evidence", type=Path, required=True, help="JSON file containing labeled evidence items"
    )
    parser.add_argument(
        "--tool",
        choices=sorted(SUPPORTED_TOOLS),
        help="Jev tool type; inferred from a top-level tool field when omitted",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("work/jev-evidence"),
        help="ignored artifact root (default: work/jev-evidence)",
    )
    parser.add_argument("--run-id", help="optional stable directory name; must be unique")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.input.read_text(encoding="utf-8"))
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        summary = record_result(
            raw, evidence, tool=args.tool, artifact_dir=args.artifacts, run_id=args.run_id
        )
    except (OSError, json.JSONDecodeError, ResultFormatError, ValueError, FileExistsError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 2 if summary["status"] in {"invalid_response", "tool_error"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
