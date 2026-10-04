# Motion search prototype

**Disposable prototype, retained outside main.** The question is whether natural-language body-motion queries can produce useful inspectable windows while exposing unavailable evidence and uncertainty. It is linked from issue#42. The answer supports an inspection workflow; it does not establish deployment accuracy or calibrated thresholds.

The HTML report has free-play threshold/review controls, guided supported-motion/missing-evidence/uncertain cases, full selected source state and raw report data. It opens by double-click. Source videos stay local; ranked results still work when a recipient has no videos. The CLI accepts repeated arbitrary `--query` values and asks typed Noul questions against frozen D3 windows. Exact-request hashes cache responses; use `--live` explicitly to send new requests. Credentials remain in the Python process, never in the report.

## Run the retained local example

From the repository root:

```sh
uv run --locked python experiments/motion-search/motion_search_prototype.py \
  --dataset work/agent-run/experiments/windows.json \
  --query 'The person repeatedly raises and lowers one or both arms.' \
  --query 'The person drinks from a red cup.' \
  --query 'The body remains almost still.' \
  --cache work/agent-run/motion-cache \
  --out work/agent-run/motion-search.html
```

The three example queries are already cached on the development machine. Add `--live` for a new query; it reads the existing `OPENROUTER_API_KEY` environment variable. `--video-base http://127.0.0.1:53731` can target the local range server; omitting it uses local file URLs. Open `work/agent-run/motion-search.html` to explore the results. The report needs no API key or network inference.

## Reproduce the research battery

The seven source pose/video clips and raw results are ignored local data. The harness uses pinned copies of `stage.py` and `nova77.py` from commit`939d1f0` through `frozen_processing.py`, so later production fixes cannot silently change the frozen study. The source copies are research dependencies, not new production modules.

```sh
uv run --locked python experiments/motion-search/research_battery.py build --limit-per-clip 12
uv run --locked python experiments/motion-search/research_battery.py retrieval
uv run --locked python experiments/motion-search/research_battery.py action-quality
uv run --locked python experiments/motion-search/research_battery.py generic-quality
uv run --locked python experiments/motion-search/research_battery.py corruption --pairs 12
uv run --locked python experiments/motion-search/research_battery.py spec-assist
uv run --locked python experiments/motion-search/analyze_results.py
uv run --locked python experiments/motion-search/make_gallery.py
```

Requests require the environment key. Exact successful research requests are cached by hash in append-only JSONL; failures stay errors. Do not combine regenerated windows with a different processing version. This branch deliberately retains the experiment source; raw pose JSON, videos, provider responses and report outputs are not committed.

## Observed behavior and limits

The initial report run made85successfulrequests and scored84windows for3queries for$0.01747326 provider-reported cost. The exact replay made0freshrequests and used85cachehits. Browser checks exercised threshold/review state, source inspection at17.533333s, and missing-appearance rejection with zero page errors. Jev Browser independently selected the red-cup query and asserted zero offered candidates; its semantic assertion matched the unavailable-evidence explanation at confidence1.0.

The corrected research battery made500successfulrequests and4,572judgments for$0.107410758; details and limitations are in the research note on main. Original/paraphrase and raw/stage rankings differ. Feature mutations are edits of digest summaries, not naturally corrupted pose arrays. The fourteen source-video strips reviewed by the host agent included seated observers and a camera operator; they are informal inspection, not human gold labels. The next production decision needs independently adjudicated labels and held-out clips. This prototype sends no automatic cuts, action claims or custom specs downstream.
