# SPDX-License-Identifier: MIT
"""PROTOTYPE: query frozen pose windows with Jev and inspect an offline HTML report."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "typesafe/jev-1.13"
ENDPOINT = "https://openrouter.ai/api/v1/systemone"
VERSION = "motion-search-prototype-1"
VIEW = "stage_world_y_up_floor_adjusted"
LEVEL = "D3_quality"


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def judge(payload, cache, live):
    """Exact-request cache; failures never become a negative motion judgment."""
    path = cache / (sha(payload) + ".json")
    if path.exists():
        record = json.loads(path.read_text())
        return {**record, "cache_hit": True}
    if not live:
        raise RuntimeError("Cache miss. Use --live to authorize an API request.")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY unavailable; no request sent")
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 -- fixed HTTPS endpoint
            answer = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Provider HTTP {error.code}; no judgment recorded") from error
    record = {
        "request_sha256": sha(payload),
        "request": payload,
        "response": answer,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 2),
        "cache_hit": False,
    }
    cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False))
    return record


def probability(record, name):
    value = record["response"]["answers"][name]
    if value.get("type") != "noul":
        raise ValueError(f"Invalid typed answer for {name}")
    number = value.get("noul")
    if isinstance(number, bool) or not isinstance(number, (float, int)):
        raise ValueError(f"Missing probability for {name}")
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Out-of-range probability for {name}")
    return number


def search(windows, queries, cache, live):
    eligibility = judge(
        {
            "model": MODEL,
            "state": {
                "queries": queries,
                "evidence_available": "Numeric 3D body joint positions and confidence, named joint angles, relative wrist/foot trajectories, normalized velocities, torso turns and periodicity. No pixels, audio, colors, props, identities or scene semantics.",
            },
            "questions": {
                f"q{i}": {
                    "type": "noul",
                    "instructions": f"Can the motion in `queries[{i}]` be judged from `evidence_available` without inventing missing facts?",
                    "criteria": {
                        "true": "The query describes observable body geometry or motion.",
                        "false": "The query requires absent appearance, props, identity, intent, health, or scene evidence.",
                    },
                }
                for i in range(len(queries))
            },
        },
        cache,
        live,
    )
    reports = [
        {"query": q, "observable": probability(eligibility, f"q{i}"), "results": [], "errors": []}
        for i, q in enumerate(queries)
    ]

    def one(window):
        state = {
            "candidate": {
                "blind_window_id": window["window_sha256"][:14],
                "source_frames": [window["source_frame_start"], window["source_frame_end"]],
                "fps": window["fps"],
                "representation": VIEW,
                "digest": window[VIEW][LEVEL],
            },
            "queries": queries,
            "task": "Use only supplied numeric evidence. Source clip names and appearance are deliberately unavailable.",
            "version": VERSION,
        }
        questions = {
            f"match{i}": {
                "type": "noul",
                "instructions": f"Does the motion in `queries[{i}]` match `candidate.digest`?",
                "criteria": {
                    "true": "The numeric evidence supports this body motion in this window.",
                    "false": "The motion is absent or the necessary evidence is insufficient.",
                },
            }
            for i in range(len(queries))
        }
        questions["unreliable"] = {
            "type": "noul",
            "instructions": "Is `candidate.digest` unreliable for motion inspection because of missing joints, unresolved sides, implausible geometry or sudden root jumps?",
            "criteria": {
                "true": "A named data defect affects inspection, or coverage is insufficient.",
                "false": "Relevant joints and geometry are usable; fast valid motion alone is not a defect.",
            },
        }
        record = judge({"model": MODEL, "state": state, "questions": questions}, cache, live)
        return window, record

    calls = [eligibility]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, w): w for w in windows}
        for future in concurrent.futures.as_completed(futures):
            w = futures[future]
            try:
                w, record = future.result()
                scores = [probability(record, f"match{i}") for i in range(len(queries))]
                risk = probability(record, "unreliable")
                calls.append(record)
                for i, score in enumerate(scores):
                    reports[i]["results"].append(
                        {
                            "window": w["window_sha256"],
                            "source": w["source_path_relative"],
                            "track": w["source_track_id"],
                            "shot": w["shot_index"],
                            "frame_start": w["source_frame_start"],
                            "frame_end": w["source_frame_end"],
                            "fps": w["fps"],
                            "score": score,
                            "risk": risk,
                            "request_sha256": record["request_sha256"],
                            "served_model": record["response"].get("model"),
                            "quality": {
                                k: w[VIEW][LEVEL].get(k)
                                for k in (
                                    "side_resolution_confident",
                                    "valid_fraction_by_joint",
                                    "confidence",
                                    "source_frame_gaps",
                                )
                            },
                        }
                    )
            except (RuntimeError, ValueError, KeyError, TypeError) as error:
                for report in reports:
                    report["errors"].append({"window": w["window_sha256"], "error": str(error)})
    for report in reports:
        report["results"].sort(key=lambda r: r["score"], reverse=True)
    fresh = [c for c in calls if not c["cache_hit"]]
    usage = {
        "requests": len(calls),
        "fresh_requests": len(fresh),
        "cache_hits": len(calls) - len(fresh),
        "input_tokens": sum(c["response"].get("usage", {}).get("input_tokens", 0) for c in fresh),
        "output_tokens": sum(c["response"].get("usage", {}).get("output_tokens", 0) for c in fresh),
        "provider_cost_usd": sum(c["response"].get("usage", {}).get("cost", 0) for c in fresh),
    }
    return {
        "version": VERSION,
        "reports": reports,
        "usage": usage,
        "eligibility_request": eligibility["request_sha256"],
        "view": VIEW,
        "digest_level": LEVEL,
    }


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Motion search experiment</title>
<style>body{font:16px/1.5 system-ui;margin:0;background:#f6f7fa;color:#152536}main{max-width:1150px;margin:auto;padding:32px}h1{font-size:32px;margin:0}p{max-width:900px}section{background:white;border:1px solid #d7deea;border-radius:12px;padding:20px;margin:20px 0}button,select,input{font:inherit}button{padding:8px 14px;border:1px solid #bac8dd;border-radius:6px;background:#edf3ff;color:#183c78;cursor:pointer;margin:4px}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}label{display:block}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #eee;padding:10px}small{color:#546476}.badge{padding:4px 8px;border-radius:4px;background:#f3e8ca}.ok{background:#deeddf}video{width:100%;max-height:420px;background:#192630}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px;font-size:12px}#notice{font-weight:600}#state{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.tabs button[aria-selected=true]{background:#254f86;color:white}</style>
<main><h1>Find a motion, inspect the evidence</h1><p>Prototype question: can a natural-language description help find inspectable body-motion windows, while preserving source time and uncertain results? This report contains real Jev judgments over frozen numeric pose summaries. Thresholds are exploratory; scores have not been calibrated against human activity labels.</p>
<section><div id="state"></div></section><section><h2>Explore the results</h2><div class="controls"><label>Motion description<select id="query"></select></label><label>Match threshold <output id="tv"></output><input id="threshold" type="range" min="0.1" max="0.95" step="0.05" value="0.8"></label><label><input id="review" type="checkbox"> Include windows needing review</label><button id="reset">Reset</button></div><p id="notice"></p><table><thead><tr><th>Source window</th><th>Match signal</th><th>Inspection</th></tr></thead><tbody id="rows"></tbody></table><p id="coverage"></p></section>
<section><h2>Guided cases</h2><div class="tabs"><button data-case="supported">Supported body motion</button><button data-case="missing">Missing appearance evidence</button><button data-case="uncertain">Uncertain windows</button></div><p id="guide"></p><button id="step">Run next step</button></section>
<section id="inspection"><h2>Inspect a source window</h2><p id="selected">Select a result to see its original track, source frames and evidence quality.</p><video id="video" controls muted preload="metadata"></video><p><small>Source video stays on this computer. A report opened without its local video files still shows the complete ranked results.</small></p><details><summary>Selected window evidence</summary><pre id="detail"></pre></details></section>
<section><h2>Recorded state and provenance</h2><p id="usage"></p><details><summary>All report data, including errors</summary><pre id="raw"></pre></details></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
// Pure decision module: evidence is immutable; only local display policy changes.
const initial=()=>({query:0,threshold:.8,includeReview:false,selected:null,scenario:'supported',step:0});
const transition=(s,a)=>a.type==='reset'?initial():{...s,...a.values};
const status=r=>r.risk>=.5||r.quality.side_resolution_confident===false?'needs review':'candidate';
const visible=(s,report)=>report.observable<.8?[]:report.results.filter(r=>r.score>=s.threshold&&(s.includeReview||status(r)==='candidate'));
let state=initial();const $=id=>document.getElementById(id);const add=(p,tag,t)=>{const e=document.createElement(tag);e.textContent=t;p.appendChild(e);return e};
data.reports.forEach((r,i)=>{let o=add($('query'),'option',r.query);o.value=i});
function dispatch(a){state=transition(state,a);render()}
function render(){const report=data.reports[state.query],hits=visible(state,report);$('query').value=state.query;$('threshold').value=state.threshold;$('tv').textContent=state.threshold.toFixed(2);$('review').checked=state.includeReview;$('state').replaceChildren();for(const [k,v] of Object.entries({'Current description':report.query,'Pose observability':report.observable.toFixed(2),'Threshold':state.threshold.toFixed(2),'Visible windows':hits.length,'Review windows included':String(state.includeReview),'Selected source':state.selected?state.selected.window.slice(0,14):'none','Walkthrough':state.scenario+' / step '+state.step})){let box=add($('state'),'div','');add(box,'small',k);add(box,'div',v)}
$('notice').textContent=report.observable<.8?'Evidence unavailable: this query needs facts absent from the pose summaries. No windows are offered.':hits.length?'Inspect candidate windows before drawing conclusions.':'No supported candidate at this threshold. Lowering the threshold exposes uncertainty.';
$('rows').replaceChildren();hits.slice(0,30).forEach(r=>{let tr=add($('rows'),'tr','');add(tr,'td',r.source+' · track '+r.track+' · '+(r.frame_start/r.fps).toFixed(2)+'-'+(r.frame_end/r.fps).toFixed(2)+'s');add(tr,'td',r.score.toFixed(2)+' · '+status(r));let cell=add(tr,'td','');let b=add(cell,'button','Inspect source');b.onclick=()=>{dispatch({values:{selected:r}});const v=$('video');if(v.getAttribute('src')!==r.video){v.src=r.video;v.onloadedmetadata=()=>{v.currentTime=r.frame_start/r.fps}}else v.currentTime=r.frame_start/r.fps;$('inspection').scrollIntoView({block:'start'})}});
$('coverage').textContent=`Showing ${Math.min(30,hits.length)} of ${hits.length} qualifying windows; ${report.results.length} scored; ${report.errors.length} request or answer failures. Failures are not negative matches.`;
if(state.selected){const r=state.selected;$('selected').textContent=`Track ${r.track}, shot ${r.shot}, original frames ${r.frame_start}-${r.frame_end} at ${r.fps} fps. Match ${r.score.toFixed(2)}; data-risk signal ${r.risk.toFixed(2)}. Reviewer must locate the identified person in the source video.`;$('detail').textContent=JSON.stringify(r,null,2)}
$('guide').textContent=state.scenario==='supported'?'Start with a body-motion query; lower the threshold, then inspect its strongest candidate.':state.scenario==='missing'?'Request a red cup. Pose evidence cannot establish object color or drinking; the eligibility judgment blocks candidate selection.':'Expose middling matches and windows with quality warnings. Match probability and evidence usability are separate signals.';
document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-selected',String(b.dataset.case===state.scenario)));
}
$('query').onchange=e=>dispatch({values:{query:Number(e.target.value),selected:null}});$('threshold').oninput=e=>dispatch({values:{threshold:Number(e.target.value)}});$('review').onchange=e=>dispatch({values:{includeReview:e.target.checked}});$('reset').onclick=()=>dispatch({type:'reset'});
document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{state=initial();dispatch({values:{scenario:b.dataset.case}})});
$('step').onclick=()=>{let q=state.scenario==='missing'?data.reports.findIndex(r=>/red cup/.test(r.query)):0;dispatch({values:{query:Math.max(q,0),threshold:state.scenario==='uncertain'?.4:state.step===0?.8:.5,includeReview:state.scenario==='uncertain',step:state.step+1,selected:null}})};
$('usage').textContent=JSON.stringify(data.usage);$('raw').textContent=JSON.stringify(data,null,2);render();
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("work/motion-search-cache"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--video-base", default="", help="Optional local HTTP root for source videos"
    )
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    windows = [w for clip in dataset["clips"] for w in clip["windows"]]
    result = search(windows, args.query, args.cache, args.live)
    result["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    result["dataset_schema"] = dataset["schema_version"]
    for report in result["reports"]:
        for row in report["results"]:
            video = Path(row["source"]).parent / "clip.mp4"
            row["video"] = (
                args.video_base.rstrip("/") + "/" + video.as_posix()
                if args.video_base
                else video.resolve().as_uri()
            )
    serialized = json.dumps(result, allow_nan=False).replace("<", "\\u003c")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.replace("__DATA__", serialized))
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "html": str(args.out.resolve()),
                "queries": [
                    {
                        "query": r["query"],
                        "observable": r["observable"],
                        "scored": len(r["results"]),
                        "errors": len(r["errors"]),
                    }
                    for r in result["reports"]
                ],
                "usage": result["usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
