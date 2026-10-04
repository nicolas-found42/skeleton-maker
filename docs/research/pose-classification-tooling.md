# Pose classification tooling

The tracked research tools make the retained pose-classification results auditable without storing source video, raw responses, or local absolute paths in Git. They are separate from the installed `skeleton-maker` command.

The artifact manifest at [`experiments/pose_classification/manifest.toml`](../../experiments/pose_classification/manifest.toml) pins the original processing commit, hashes of the frozen processing modules, and hashes for the six local inputs: the window dataset, source manifest, and four classification result files. Place those files under `work/agent-run/experiments/` or pass their containing directory explicitly. The original source clips and generated study data remain local and ignored by Git.

Run the offline synthetic checks and verify a local retained dataset with:

```sh
uv run --no-sync python -m experiments.pose_classification.research_battery preflight
uv run --no-sync python -m experiments.pose_classification.research_battery verify \
  --artifacts work/agent-run/experiments
uv run --no-sync python -m experiments.pose_classification.research_battery summary \
  --artifacts work/agent-run/experiments
```

`preflight` checks a stationary and bouncing synthetic root, the median-centered hip-height feature, camera y-down sign conversion, body-scale normalization, a missing wrist, a source-frame gap, rejection of non-finite pose values and malformed TypeSafe responses, and the request/response schemas. `verify` checks every pinned artifact SHA256 and parses each stored request/response pair before returning record counts. `summary` validates the same inputs first, then reports fixed choice-label counts by experiment and question, along with total requests, judgments, provider-reported token use, and cost. The retained run sums to 248 requests, 1,548 judgments, 1,261,833 input tokens, 37,710 output tokens, and $0.052996986 provider-reported cost.

To reconstruct the exact stored TypeSafe calls and answers without credentials or network access:

```sh
uv run --no-sync python -m experiments.pose_classification.research_battery replay \
  --artifacts work/agent-run/experiments \
  --output work/agent-run/research-replay.jsonl
```

The exported JSONL contains each original `model`, `state`, `questions`, request hash, and validated response. The four input files contain 248 successful result records. Replay validates all source records and builds the complete output in memory before replacing the target atomically.

The frozen processor can also be run directly on a local NIM `pose.json` JSONL file, without an API key:

```sh
uv run --no-sync python -m experiments.pose_classification.research_battery frozen-stage \
  --pose-json in/example.skeleton/pose.json \
  --output work/agent-run/example-frozen-stage.json
```

It uses the `stage` and `nova77` implementations pinned to commit `939d1f0ed30d4f47ed8d3f31b53cda94f481b45d`. Output includes the stage metadata and payload and should stay in ignored `work/` storage.

For a new local numeric pose window, the fixed classifier accepts JSON with `source_frames`, `fps`, `coordinate_convention` (`raw_camera_y_down` or `stage_y_up`), and a `joints` map containing one 3D vector or `null` per frame. It calculates median-centered hip height, normalized root and joint speeds, knee angles, wrist height and reversals, gaps, missing-data rates, and coverage. It only sends these numeric measurements with fixed action and quality questions. Unobserved measurements are `null` with their coverage shown; the Choice instruction directs Jev to `other_or_ambiguous` when no available feature supports a specific label. A result is cached by hash of the exact typed request and can be replayed without a key.

The live request follows the TypeSafe [System One HTTP schema](https://docs.typesafe.ai/api): a request has `model`, `state`, and typed `questions`; responses have `model`, `answers`, and `usage`. Choice probabilities must cover every declared option and sum to one; Noul values are probabilities from zero to one. The OpenRouter [TypeSafe SDK compatibility guide](https://openrouter.ai/docs/guides/community/typesafe-sdk) documents the System One endpoint `https://openrouter.ai/api/v1/systemone` and the additional `id`, `provider`, and `usage.cost` fields. Live `classify` requires `OPENROUTER_API_KEY`; no live calls are part of `preflight`, `verify`, `summary`, or `replay`.

These model labels and synthetic corruptions are exploratory judgments, not human-grounded activity labels, accuracy estimates, or calibrated quality guarantees. See [`jev-real-pose-study.md`](jev-real-pose-study.md) for the retained study counts, source provenance, and interpretation limits.
