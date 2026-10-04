# Jev evidence reporting

Use `scripts.jev_report.record_result` after a Jev MCP call when the full
response belongs in local evidence but only a compact status belongs in the
agent conversation. The helper archives the raw MCP envelope, unwrapped tool
payload, labeled evidence, and a deterministic summary under an ignored
artifact directory. It does not call Jev, judge correctness, or turn a review
into approval.

```python
from scripts.jev_report import record_result

summary = record_result(
    live_jev_result,
    [{"id": "offline-tests", "text": test_output}],
    tool="gate",
    artifact_dir="work/jev-evidence",
)
print(summary)
```

The returned summary includes the artifact path, tool action, status, claim
verdict counts, each contradicted/unsupported/unknown/review item, warning and
error text, limiting review rubrics, confidence, safe-to-apply/composite values,
reported usage, and evidence IDs. A gate with verified claims and action
`review` remains `review`; the reporter applies no new threshold. `summarize`
raises `ResultFormatError` for unknown or malformed shapes. `record_result`
archives malformed responses as explicit `invalid_response` failures for
debugging, while a tool's own `invalid_response` status remains explicit. The
CLI exits with status 2 for invalid responses and MCP tool errors; completed
reviews and escalations remain successful reports. The archive keeps the
original evidence input when its normalized labeled form is malformed. Verify
results with no aggregate action derive `escalate` if any individual result
requests escalation, otherwise `review` if any result requests review, or `auto`
only when every result explicitly says `auto`;
`action_source` marks this as a derivation. Per-result confidence and action
remain visible in the summary. For a direct CLI call, write the native Jev/MCP
result and labeled evidence as JSON files, then run:

```bash
uv run python scripts/jev_report.py \
  --input work/jev-input.json \
  --evidence work/jev-evidence-input.json \
  --tool gate \
  --artifacts work/jev-evidence
```

For dependent commands, import `Action`, `run_checked`, and
`CheckedCommandError` from `scripts.run_checked`. Each action is an argv list,
not a shell string. The runner stops at the first failing subprocess, replays
its output, removes the staged files, and raises with the action name and exit
status. It publishes logs and a manifest through one directory rename only
after every action succeeds. The destination must be new. Its CLI accepts a
JSON plan with an `actions` array containing `{ "name": "...", "argv": [...] }`
objects and requires a new `--artifacts` destination. The manifest stores action
names and log paths, not argv values, so command-line secrets are not copied
into another artifact.

These helpers are local orchestration aids. A compact summary is not a
verification: inspect raw responses and cited evidence when a decision matters.
Jev's statuses, probabilities, warnings, and token usage are preserved as
returned; the model remains an advisory source and does not enforce project
policy. Store private or sensitive inputs only where the repository permits;
the default artifact path is under gitignored `work/`.
