# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Analyze frozen Jev battery results with explicit mechanical-reference caveats."""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
from collections import Counter
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
HERE = SOURCE.parents[1] / "work" / "agent-run" / "experiments"
SPEC = importlib.util.spec_from_file_location("battery", SOURCE / "research_battery.py")
battery = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("Research module loader unavailable")
SPEC.loader.exec_module(battery)


def rows(name):
    p = HERE / f"{name}_v1.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def ans(r, q):
    return r.get("response", {}).get("answers", {}).get(q, {})


def p_noul(r, q):
    a = ans(r, q)
    return a.get("noul") if a.get("type") == "noul" else None


def report_requests(rs):
    good = [r for r in rs if r["status"] == "ok"]
    usages = [r.get("response", {}).get("usage", {}) for r in good]
    costs = [x.get("cost") for x in usages if isinstance(x.get("cost"), (int, float))]
    lats = [r["latency_ms"] for r in good if r.get("latency_ms") is not None]
    return {
        "requests": len(rs),
        "success": len(good),
        "failures": len(rs) - len(good),
        "judgments_successful": sum(r.get("question_count", 0) for r in good),
        "input_tokens": sum(int(x.get("input_tokens", 0) or 0) for x in usages),
        "output_tokens": sum(int(x.get("output_tokens", 0) or 0) for x in usages),
        "reported_cost_usd": sum(costs) if costs else None,
        "median_latency_ms": statistics.median(lats) if lats else None,
        "p95_latency_ms": float(__import__("numpy").percentile(lats, 95)) if lats else None,
        "returned_models": sorted({r.get("response", {}).get("model") for r in good}),
        "http_errors": dict(Counter(str(r.get("http_status")) for r in rs if r["status"] != "ok")),
    }


def binary_metrics(pred, labels):
    pairs = [
        (float(p), int(y))
        for p, y in zip(pred, labels, strict=False)
        if p is not None and y is not None
    ]
    if not pairs:
        return {"n": 0}
    ps, ys = zip(*pairs, strict=False)
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    mean_p = sum(ps) / len(ps)
    prevalence = sum(ys) / len(ys)

    def conf(t):
        tp = sum(p >= t and y for p, y in pairs)
        fp = sum(p >= t and not y for p, y in pairs)
        fn = sum(p < t and y for p, y in pairs)
        tn = sum(p < t and not y for p, y in pairs)
        return {
            "threshold": t,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    auc = (
        sum((1 if x > n else 0.5 if x == n else 0) for x in pos for n in neg)
        / (len(pos) * len(neg))
        if pos and neg
        else None
    )
    bins = []
    for lo in [0, 0.2, 0.4, 0.6, 0.8]:
        grp = [(p, y) for p, y in pairs if lo <= p < lo + 0.2 or (lo == 0.8 and p == 1)]
        if grp:
            bins.append(
                {
                    "range": f"{lo:.1f}-{lo + 0.2:.1f}",
                    "n": len(grp),
                    "mean_probability": sum(p for p, _ in grp) / len(grp),
                    "observed_proxy_rate": sum(y for _, y in grp) / len(grp),
                }
            )
    return {
        "n": len(pairs),
        "proxy_prevalence": prevalence,
        "mean_probability": mean_p,
        "brier_vs_mechanical_proxy": brier,
        "auc_vs_mechanical_proxy": auc,
        "thresholds": [conf(t) for t in (0.5, 0.8, 0.9)],
        "calibration_bins_vs_mechanical_proxy": bins,
    }


def spearman(a, b):
    def rank(x):
        s = sorted((v, i) for i, v in enumerate(x))
        out = [0.0] * len(x)
        k = 0
        while k < len(s):
            j = k + 1
            while j < len(s) and s[j][0] == s[k][0]:
                j += 1
            av = (k + j - 1) / 2 + 1
            for _, i in s[k:j]:
                out[i] = av
            k = j
        return out

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=False))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va * vb else None


def main():
    retrieval = rows("retrieval_digests")
    action = rows("action_quality")
    corruption = rows("injected_corruption")
    specs = rows("spec_assist")
    generic = rows("quality_generic")
    ds = json.loads((HERE / "windows.json").read_text())
    windows = {w["window_sha256"]: w for c in ds["clips"] for w in c["windows"]}
    source = json.loads((HERE / "source.json").read_text())
    source_idx = {x["window_sha256"]: x for x in source["selected_windows"]}
    metrics = {
        "schema": "jev-real-pose-battery-analysis-v1",
        "data": {
            "selected_windows": len(windows),
            "clips": len(ds["clips"]),
            "frames_per_window": ds["window_frames"],
            "model_requested": ds["model"],
            "prompt_version": ds["schema_version"],
            "human_labels": False,
        },
        "operations": {},
    }
    for name, rs in [
        ("retrieval", retrieval),
        ("action_quality", action),
        ("injected_corruption", corruption),
        ("spec_assist", specs),
        ("generic_quality", generic),
    ]:
        metrics["operations"][name] = report_requests(rs)
    metrics["operations"]["all_v1"] = {
        k: sum(
            metrics["operations"][n][k] or 0
            for n in (
                "retrieval",
                "action_quality",
                "injected_corruption",
                "spec_assist",
                "generic_quality",
            )
        )
        for k in (
            "requests",
            "success",
            "failures",
            "judgments_successful",
            "input_tokens",
            "output_tokens",
            "reported_cost_usd",
        )
    }
    metrics["operations"]["all_v1"]["median_latency_ms"] = "see per-experiment distributions"

    # Motion retrieval against explicitly defined geometry rules only. Red-cup is no-evidence.
    retrieval_by = {}
    for r in retrieval:
        for qid, v in r.get("response", {}).get("answers", {}).items():
            retrieval_by[(r["window_sha256"], qid)] = v.get("noul")
    proxy = {}
    for q in battery.QUERY_SPECS:
        qs = []
        ys = []
        for h, w in windows.items():
            val = battery.mc_label(w, q, "stage_world_y_up_floor_adjusted", "D3_quality")[
                "mechanical_reference"
            ]
            if val is None:
                continue
            key = (h, f"{q['id']}__original__stage_D3")
            if key in retrieval_by:
                qs.append(retrieval_by[key])
                ys.append(val)
        proxy[q["id"]] = binary_metrics(qs, ys)
    metrics["motion_query_vs_mechanical_references_not_human_truth"] = proxy
    {k: v for k, v in proxy.items() if k == "red_cup"}
    red_q = [
        retrieval_by[(h, qid)]
        for h in windows
        for qid in ("red_cup__original__stage_D3",)
        if (h, qid) in retrieval_by
    ]
    metrics["unobservable_query_red_cup_no_evidence"] = {
        "n": len(red_q),
        "mean_probability": sum(red_q) / len(red_q) if red_q else None,
        "hits_ge_0_5": sum(p >= 0.5 for p in red_q),
        "hits_ge_0_8": sum(p >= 0.8 for p in red_q),
        "hits_ge_0_9": sum(p >= 0.9 for p in red_q),
        "max_probability": max(red_q) if red_q else None,
        "interpretation": "structural no-evidence control; not a visual evaluation of cup presence",
    }

    # Raw camera D3 vs stage D3 paired score/decision changes.
    pair = {}
    for q in battery.QUERY_SPECS:
        diffs = []
        flips = []
        for h in windows:
            a = retrieval_by.get((h, f"{q['id']}__original__raw_D3"))
            b = retrieval_by.get((h, f"{q['id']}__original__stage_D3"))
            if a is not None and b is not None:
                diffs.append(b - a)
                flips.append((a >= 0.5) != (b >= 0.5))
        pair[q["id"]] = {
            "n": len(diffs),
            "mean_stage_minus_raw": sum(diffs) / len(diffs) if diffs else None,
            "mean_absolute_difference": sum(abs(x) for x in diffs) / len(diffs) if diffs else None,
            "threshold_flip_count_ge_0_5": sum(flips),
            "threshold_flip_rate": sum(flips) / len(flips) if flips else None,
        }
    metrics["paired_raw_camera_D3_vs_stage_D3"] = pair
    d0d3 = {}
    for q in battery.QUERY_SPECS:
        diffs = []
        flips = []
        for h in windows:
            a = retrieval_by.get((h, f"{q['id']}__original__stage_D0"))
            b = retrieval_by.get((h, f"{q['id']}__original__stage_D3"))
            if a is not None and b is not None:
                diffs.append(b - a)
                flips.append((a >= 0.5) != (b >= 0.5))
        d0d3[q["id"]] = {
            "n": len(diffs),
            "mean_absolute_difference": sum(abs(x) for x in diffs) / len(diffs) if diffs else None,
            "threshold_flip_count_ge_0_5": sum(flips),
            "threshold_flip_rate": sum(flips) / len(flips) if flips else None,
        }
    metrics["paired_stage_D0_vs_D3"] = d0d3
    parap = {}
    for q in battery.QUERY_SPECS:
        original = []
        p1 = []
        diffs = []
        flips = []
        for h in windows:
            a = retrieval_by.get((h, f"{q['id']}__original__stage_D3"))
            b = retrieval_by.get((h, f"{q['id']}__paraphrase_1__stage_D3"))
            if a is not None and b is not None:
                original.append(a)
                p1.append(b)
                diffs.append(b - a)
                flips.append((a >= 0.5) != (b >= 0.5))
        parap[q["id"]] = {
            "n": len(diffs),
            "spearman_rank_agreement": spearman(original, p1) if len(diffs) > 1 else None,
            "mean_absolute_score_difference": sum(abs(x) for x in diffs) / len(diffs)
            if diffs
            else None,
            "threshold_flip_count_ge_0_5": sum(flips),
            "threshold_flip_rate": sum(flips) / len(flips) if flips else None,
        }
    metrics["paired_original_vs_paraphrase_1_stage_D3"] = parap

    # Choice vs atomic action proxies and composition (fixed code priorities).
    confusion = Counter()
    atom_con = Counter()
    exact = 0
    for r in action:
        h = r["window_sha256"]
        w = windows[h]
        got = ans(r, "action_stage").get("choice")
        ref = battery.action_reference(w, "stage_world_y_up_floor_adjusted")
        confusion[(ref, got)] += 1
        exact += got == ref
        aa = {
            k: p_noul(r, k)
            for k in (
                "action_deep_bend",
                "action_arms_overhead",
                "action_repetitive_upper",
                "action_large_root_motion",
                "action_mostly_still",
            )
        }
        # Prespecified deterministic composition priority, applied without fitting on outcomes.
        if aa["action_deep_bend"] is not None and aa["action_deep_bend"] >= 0.7:
            comp = "deep_bend"
        elif aa["action_arms_overhead"] is not None and aa["action_arms_overhead"] >= 0.7:
            comp = "arms_overhead"
        elif aa["action_repetitive_upper"] is not None and aa["action_repetitive_upper"] >= 0.7:
            comp = "repetitive_upper_body"
        elif aa["action_large_root_motion"] is not None and aa["action_large_root_motion"] >= 0.7:
            comp = "large_root_motion"
        elif aa["action_mostly_still"] is not None and aa["action_mostly_still"] >= 0.7:
            comp = "mostly_still"
        else:
            comp = "other_or_ambiguous"
        atom_con[(ref, comp)] += 1
    [battery.action_reference(w, "stage_world_y_up_floor_adjusted") for w in windows.values()]
    # Choice composition comparison to the same geometric proxy reference; not human semantic quality.
    metrics["action_choice_vs_mechanical_priority_proxy"] = {
        "n": len(action),
        "proxy_exact_match": exact,
        "proxy_accuracy": exact / len(action) if action else None,
        "confusion_ref_rows_choice_columns": {f"{a}|{b}": n for (a, b), n in confusion.items()},
        "fixed_atomic_threshold_composition_confusion_ref_rows_composed_columns": {
            f"{a}|{b}": n for (a, b), n in atom_con.items()
        },
        "note": "reference applies code thresholds in action_reference; not a human action label; atomic thresholds/priority are exploratory, fixed before metric computation",
    }

    # Specific per-mode vs generic detector, stratified by clean proxy flags.
    qmetrics = {}
    clean_any = []
    generic_clean = []
    generic_by = {r["window_sha256"]: p_noul(r, "generic_unreliable") for r in generic}
    qby = {r["window_sha256"]: r for r in action}
    for mode in battery.QUALITY_QUESTIONS:
        pp = []
        yy = []
        for h, w in windows.items():
            pp.append(p_noul(qby[h], "quality_" + mode))
            yy.append(battery.ref_quality(w, mode))
        qmetrics[mode] = binary_metrics(pp, yy)
    for h, w in windows.items():
        clean = not any(battery.ref_quality(w, m) for m in battery.QUALITY_QUESTIONS)
        if clean:
            clean_any.append(h)
            generic_clean.append(generic_by.get(h))
    metrics["specific_quality_detectors_vs_mechanical_references_not_human_truth"] = qmetrics
    metrics["generic_unreliable_vs_union_quality_proxy"] = {
        "n": len(windows),
        "any_quality_proxy_positive": sum(
            not all(not battery.ref_quality(w, m) for m in battery.QUALITY_QUESTIONS)
            for w in windows.values()
        ),
        "brier_vs_union_proxy": binary_metrics(
            [generic_by.get(h) for h in windows],
            [
                any(battery.ref_quality(w, m) for m in battery.QUALITY_QUESTIONS)
                for w in windows.values()
            ],
        )["brier_vs_mechanical_proxy"],
        "rule_clean_n": len(clean_any),
        "generic_false_alarm_ge_0_5_on_rule_clean": sum(
            p is not None and p >= 0.5 for p in generic_clean
        ),
        "generic_mean_probability_rule_clean": sum(p for p in generic_clean if p is not None)
        / sum(p is not None for p in generic_clean)
        if any(p is not None for p in generic_clean)
        else None,
        "specific_detector_flags": {
            "quality_" + m: sum((p_noul(qby[h], "quality_" + m) or 0) >= 0.5 for h in clean_any)
            for m in battery.QUALITY_QUESTIONS
        },
        "caveat": "quality rules are transparent mechanical references and may have false positives/negatives; not adjudicated corruption labels",
    }

    # Reconstruct injection mode from exact request payload hash; no network calls.
    {w["window_sha256"]: w for w in windows.values()}
    injmap = {}
    chosen = list(windows.values())[:: max(1, len(windows) // 12)][:12]
    for w in chosen:
        for mode in battery.QUALITY_QUESTIONS:
            cor = battery.inject(w, mode)
            state = battery.model_state(cor, "stage_world_y_up_floor_adjusted", "D3_quality")
            questions = {
                "detector_" + k: {
                    "type": "noul",
                    "instructions": {"question": text, "evidence": "candidate.digest"},
                    "criteria": {
                        "true": "The evidence indicates this failure is present.",
                        "false": "The evidence does not indicate this failure.",
                    },
                }
                for k, text in battery.QUALITY_QUESTIONS.items()
            }
            injmap[
                battery.jhash({"model": battery.MODEL, "state": state, "questions": questions})
            ] = (w["window_sha256"], mode)
    injstats = {
        m: {
            "n": 0,
            "hit_at_0_5": 0,
            "mean_intended_probability": 0.0,
            "mean_other_probability": 0.0,
            "cross_flags": 0,
        }
        for m in battery.QUALITY_QUESTIONS
    }
    bad = []
    for r in corruption:
        pair = injmap.get(r["request_sha256"])
        if pair is None:
            bad.append(r["request_sha256"])
            continue
        h, mode = pair
        ps = {k: p_noul(r, "detector_" + k) for k in battery.QUALITY_QUESTIONS}
        st = injstats[mode]
        st["n"] += 1
        intended = ps[mode]
        others = [v for k, v in ps.items() if k != mode and v is not None]
        st["mean_intended_probability"] += intended or 0
        st["mean_other_probability"] += sum(others) / len(others) if others else 0
        st["hit_at_0_5"] += int(intended is not None and intended >= 0.5)
        st["cross_flags"] += sum(v is not None and v >= 0.5 for v in others)
    for st in injstats.values():
        if st["n"]:
            st["mean_intended_probability"] /= st["n"]
            st["mean_other_probability"] /= st["n"]
            st["intended_detection_rate_ge_0_5"] = st["hit_at_0_5"] / st["n"]
    metrics["feature_level_injections_on_real_window_digests_not_raw_pose_corruption"] = {
        "per_mode": injstats,
        "unmapped_records": len(bad),
        "n_windows_with_repeated_modes": 12,
        "n_modes": 6,
        "caveat": "only compact D3 feature values were mutated; source NIM keypoints/pose.json and stage output were untouched; this tests stated detector response to declared feature edits, not robustness to naturally occurring pose corruption",
    }

    # Request-to-existing-spec bounded fixture answers.
    sm = []
    for r in specs:
        sid = r["experiment_item_id"]
        fixture = next(x for x in battery.SPEC_REQUESTS if x["id"] == sid)
        sm.append(
            {
                "fixture_id": sid,
                "authored_target": fixture["target"],
                "selected": ans(r, "best_character").get("choice"),
                "fits_existing_probability": p_noul(r, "fits_existing_scope"),
                "requires_new_capability_probability": p_noul(r, "requires_new_capability"),
            }
        )
    metrics["spec_assist_authored_fixture_summary"] = {
        "n": len(sm),
        "authored_catalog_targets": sum(x["authored_target"] != "none" for x in sm),
        "choice_exact_match_to_authored_target": sum(
            x["selected"] == x["authored_target"] for x in sm
        ),
        "out_of_catalog_rejected": sum(
            x["authored_target"] == "none" and x["selected"] == "none" for x in sm
        ),
        "fixture_rows": sm,
        "caveat": "targets were authored alongside examples; these are smoke fixtures, not independent user requests or a validated spec-assistance benchmark",
    }

    metrics["provenance"] = {
        "selected_window_hashes": len(source_idx),
        "local_source_file_hashes": {
            x["local_path"]: x["source_file_sha256"] for x in source["sources"]
        },
        "request_results_have_request_state_question_sha256": all(
            all(k in r for k in ("request_sha256", "state_sha256", "questions_sha256"))
            for rs in (retrieval, action, corruption, specs, generic)
            for r in rs
        ),
        "no_media_copied_into_model_state": True,
    }
    out = HERE / "metrics_v1.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "path": str(out),
                "operations": metrics["operations"],
                "all_v1": metrics["operations"]["all_v1"],
                "red_cup": metrics["unobservable_query_red_cup_no_evidence"],
                "spec_assist": metrics["spec_assist_authored_fixture_summary"],
                "injections": metrics[
                    "feature_level_injections_on_real_window_digests_not_raw_pose_corruption"
                ]["per_mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
