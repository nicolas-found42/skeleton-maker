# Jev real-pose classification and quality study

Run date: 2026-10-04. The study uses 84 selected windows from seven local 600-frame clips, with 60 frames per window. Source clips, numeric digests and raw responses stay in ignored `work/agent-run/experiments/`. Request state omits clip labels, local paths, activity names and source track IDs; local provenance joins those fields for inspection.

## Retained scope

The maintainer withdrew natural-language motion search on 2026-10-04. [Draft PR#44](https://github.com/nicolas-found42/skeleton-maker/pull/44) was closed without merging, [issue#42](https://github.com/nicolas-found42/skeleton-maker/issues/42) was closed as not planned, and the search branch was deleted. Local query-matching and ranking code, the search CLI, interactive query report, gallery generator and query cache were removed. The managed prototype worktree was archived for recovery. Historical response data and Git history remain research records; they are not an active search mechanism.

Action classifications, independent motion-property probabilities, pose-quality judgments, frozen numeric digests and source-frame provenance are retained. There is no natural-language motion-search command in the product. The character rendering feature shipped separately in [PR#33](https://github.com/nicolas-found42/skeleton-maker/pull/33).

## Measurements

The retained classification, quality and bounded character-catalog fixture results comprise **1,548 judgments in 248 successful requests**, with no failed v1 requests. OpenRouter reported **1,261,833 input tokens**, **37,710 output tokens** and **$0.052996986** total cost for these four experiments. The requested model was `typesafe/jev-1.13`; responses identified `typesafe/jev-1.13-20260917`. These totals exclude the withdrawn retrieval experiment and invalid v0 pilots.

| Experiment | Requests | Judgments | Provider-reported cost, USD |
| --- | ---: | ---: | ---: |
| Closed action Choice, five independent properties, six quality questions | 84 | 1,008 | 0.020401836 |
| One generic reliability question | 84 | 84 | 0.016849140 |
| Six quality questions on 72 edited digest copies | 72 | 432 | 0.015402114 |
| Bounded built-in-character catalog fixtures | 8 | 24 | 0.000343896 |
| Total | 248 | 1,548 | 0.052996986 |

The local `classification_metrics_v1.json` contains these aggregates, quality comparisons, corruption outcomes and catalog-fixture responses. Raw records retain typed Noul/Choice answers, full distributions, request/state/question SHA256 hashes, latency, model version and usage. The closed action catalog is `deep_bend`, `arms_overhead`, `repetitive_upper_body`, `large_root_motion`, `mostly_still`, and `other_or_ambiguous`; those are coarse digest labels, not sport or activity identities.

Some observations from this dataset:

- Specific quality checks flagged 0/28 rule-clean windows at probability ≥0.5 across the six named questions. The generic question also flagged 0/28 of these windows, with mean score 0.154. This does not establish clean-data accuracy: the rules are not adjudicated and most positive failure types are rare or absent. Generic Brier score against the union-of-rules proxy was 0.310.
- Six explicit summary-feature edits were applied to 12 sampled real-window digest copies per mode. At p≥0.5, the intended detector flagged 12/12 missing-key-data copies, 8/12 root-jump copies, 3/12 bone-instability copies, 12/12 side-ambiguity copies, 12/12 body-scale-outlier copies and 12/12 temporal-gap copies. Only digest summaries were edited; raw pose arrays and original stage results stayed unchanged. This tests response to explicit edits, not robustness to natural tracking failures.
- The character-catalog fixture selected all eight co-authored targets, including two `none` cases. Examples and targets were authored together, so this is a smoke check, not an independent request benchmark or evidence of character-spec quality. The retained fixture does not accept free-text user input or search motion windows.
- The host agent inspected fourteen timestamp strips with three crops each, covering all seven clips. Selected tracks included a seated camera operator and observers with occluded legs. Clip genre therefore cannot label a selected person's activity. This informal inspection is neither human nor blinded ground truth; three stills cannot establish repeated-motion counts.

## Local artifacts and reproduction

The classification-only research harness remains local at `work/agent-run/experiments/research_battery.py`. Its commands accept a fixed experiment catalog; the retrieval command, query definitions and query-scoring function were removed. Source clips and responses are not committed. With the repository environment and `OPENROUTER_API_KEY` available for fresh live judgments, the bounded commands are:

```sh
uv run --locked python work/agent-run/experiments/research_battery.py --help
uv run --locked python work/agent-run/experiments/research_battery.py action-quality
uv run --locked python work/agent-run/experiments/research_battery.py generic-quality
uv run --locked python work/agent-run/experiments/research_battery.py corruption --pairs 12
uv run --locked python work/agent-run/experiments/research_battery.py spec-assist
uv run --locked python work/agent-run/experiments/research_battery.py summary action_quality
```

Successful exact requests are cached in their JSONL result files; matching replay does not submit new requests. Do not print or save the API key. This local harness and its inputs are not a distributed benchmark or a production command.

Frozen windows are in `windows.json`, local source joins in `source.json`, and source hashes in `provenance_v1.json`. The generating processing version was commit `939d1f0`; local `frozen_stage.py`, `frozen_nova77.py` and `frozen_processing.py` preserve that version. Later production fixes are not silently folded into the study. Do not rebuild windows with different processing and combine them with these response files. Archived invalid v0 pilots are excluded from v1 counts.

## TypeSafe contract used

TypeSafe's [Python quick start](https://docs.typesafe.ai/sdk/python.md) and [synchronous client reference](https://docs.typesafe.ai/sdk/python/api/clients/sync.md) document `TypeSafeClient(...).system_one(state=..., questions=..., model=...)`. [System One](https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md) supports independent typed questions against one state. [Noul](https://docs.typesafe.ai/primitives/noul.md) returns a yes-probability; [Choice](https://docs.typesafe.ai/primitives/choice.md) returns a choice distribution.

The harness used the HTTP contract documented by [OpenRouter's TypeSafe SDK guide](https://openrouter.ai/docs/guides/community/typesafe-sdk): OpenRouter credentials, the `typesafe/jev-1.13` namespace, and `/v1/systemone`. The project environment did not have the TypeSafe SDK installed, so the harness made the equivalent request directly. Responses included usage and provider-reported cost.

## Limits and next steps

These are mechanical-reference comparisons and co-authored smoke fixtures, not human-grounded accuracy or probability calibration. The sample is a selected 12-window slice per clip; quality positives are scarce, thresholds were not validated on held-out clips, and numeric pose summaries lose visual evidence. Jev judgments support inspection and experiment selection; deterministic geometry, validation and execution remain in code.

For further classification work, label a stratified sample of action/property facts and quality faults with blinded reviewers, adjudicate disagreements, then evaluate held-out clips. Add naturally observed tracking failures before treating feature-edit checks as quality validation. Keep file names and reviewer labels out of model state. Unmeasured stage-construction scaling concerns remain in [issue#43](https://github.com/nicolas-found42/skeleton-maker/issues/43) for benchmarking before optimization.
