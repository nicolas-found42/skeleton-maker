# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Build a local, text-only review gallery from frozen model outputs and provenance."""

import html
import importlib.util
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
HERE = SOURCE.parents[1] / "work" / "agent-run" / "experiments"
spec = importlib.util.spec_from_file_location("battery_gallery", SOURCE / "research_battery.py")
battery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(battery)
ds = json.loads((HERE / "windows.json").read_text())
src = json.loads((HERE / "source.json").read_text())
metrics = json.loads((HERE / "metrics_v1.json").read_text())
ws = {w["window_sha256"]: w for c in ds["clips"] for w in c["windows"]}
prov = {w["window_sha256"]: w for w in src["selected_windows"]}
retr = [json.loads(x) for x in (HERE / "retrieval_digests_v1.jsonl").read_text().splitlines()]
actions = [json.loads(x) for x in (HERE / "action_quality_v1.jsonl").read_text().splitlines()]
generic = [json.loads(x) for x in (HERE / "quality_generic_v1.jsonl").read_text().splitlines()]


def answers(rows):
    out = {}
    for r in rows:
        for q, a in r.get("response", {}).get("answers", {}).items():
            out[(r["window_sha256"], q)] = a
    return out


ra, aa, ga = answers(retr), answers(actions), answers(generic)


def esc(x):
    return html.escape(str(x))


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


queries = [
    ("hands_over_head", "Hands above head"),
    ("arms_raised", "Repeated arm raising"),
    ("deep_knee_bend", "Deep knee bend"),
    ("body_bounce", "Hip bounce"),
    ("mostly_still", "Mostly still"),
    ("red_cup", "Red cup (unobservable)"),
]
lines = [
    '<!doctype html><meta charset="utf-8"><title>Jev pose battery v1 review</title><style>body{font:14px system-ui;margin:2rem;max-width:1200px;color:#222}table{border-collapse:collapse;width:100%;margin:1rem 0 2rem}th,td{border:1px solid #ccc;padding:.4rem;text-align:left}th{background:#eee}small{color:#555}.tag{font-family:monospace}</style>',
    "<h1>Jev real-pose battery — local review gallery</h1>",
    "<p>84 deterministically sampled 2-second windows across seven local pose files. This page joins model output to local source frame/track provenance for human inspection. It contains no video or keypoint media. It is not a ground-truth accuracy dashboard.</p>",
]
lines.append(
    "<h2>Run summary</h2><p>500/500 requests succeeded; 4,572 submitted judgments; OpenRouter reported $0.107410758; requested model <code>typesafe/jev-1.13</code>, returned <code>typesafe/jev-1.13-20260917</code>. Interpret mechanical proxy metrics as rule agreement, not human truth.</p>"
)
lines.append(
    "<h2>Query score extremes, stage D3</h2><p>Scores are Jev Noul probabilities. Local source names and frames are shown only for review; they were not sent in model state.</p>"
)
for q, label in queries:
    lines.append(
        f"<h3>{esc(label)}</h3><table><thead><tr><th>Window</th><th>Score</th><th>Proxy</th><th>Local source</th><th>Track/shot</th><th>Frames</th></tr></thead><tbody>"
    )
    vals = []
    for h, w in ws.items():
        a = ra.get((h, f"{q}__original__stage_D3"), {})
        p = a.get("noul")
        if p is None:
            continue
        ref = next(
            (
                m
                for m in metrics.get("motion_query_vs_mechanical_references_not_human_truth", {})
                .get(q, {})
                .items()
            ),
            None,
        )
        rule = None
        try:
            qspec = next(z for z in battery.QUERY_SPECS if z["id"] == q)
            rule = battery.mc_label(w, qspec, "stage_world_y_up_floor_adjusted", "D3_quality")[
                "mechanical_reference"
            ]
        except Exception:
            rule = None
        vals.append((p, h, rule))
    for p, h, rule in (
        sorted(vals, key=lambda z: z[0], reverse=True)[:5] + sorted(vals, key=lambda z: z[0])[:5]
    ):
        x = prov[h]
        lines.append(
            f'<tr><td class="tag">{h[:14]}</td><td>{p:.2f}</td><td>{"no evidence rule" if rule is None else str(rule).lower()}</td><td>{esc(x["local_path"])}</td><td>{esc(x["source_track_id"])}/{esc(x["shot_index"])}</td><td>{x["source_frame_start"]}-{x["source_frame_end"]}</td></tr>'
        )
    lines.append("</tbody></table>")
lines.append(
    "<h2>Quality flags and generic reliability scores</h2><table><thead><tr><th>Window</th><th>Generic unreliable</th><th>Specific flags at ≥0.5</th><th>Local source / frames</th></tr></thead><tbody>"
)
for h, _w in ws.items():
    g = ga.get((h, "generic_unreliable"), {}).get("noul")
    flags = []
    for q in (
        "missing_key_data",
        "root_jump",
        "bone_scale_instability",
        "side_ambiguity",
        "body_scale_outlier",
        "temporal_gap",
    ):
        p = aa.get((h, "quality_" + q), {}).get("noul")
        if p is not None and p >= 0.5:
            flags.append(f"{q}:{p:.2f}")
    if (g or 0) >= 0.5 or flags:
        x = prov[h]
        lines.append(
            f'<tr><td class="tag">{h[:14]}</td><td>{fmt(g)}</td><td>{esc(", ".join(flags) or "—")}</td><td>{esc(x["local_path"])} [{x["source_frame_start"]}-{x["source_frame_end"]}]</td></tr>'
        )
lines.append(
    "</tbody></table><p><small>High scores are inspection leads. Human visual review is still needed before claiming semantic relevance or tracking-failure accuracy.</small></p>"
)
(HERE / "review_gallery_v1.html").write_text("\n".join(lines) + "\n")
print(HERE / "review_gallery_v1.html")
