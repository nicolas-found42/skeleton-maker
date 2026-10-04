---
name: Bug report
about: Something does not work as documented
title: "fix: "
labels: needs-triage
---

## Summary

<!-- Component + trigger + observable failure, in one or two sentences. -->

## Impact

<!-- User task blocked, severity and proposed priority with reasons, occurrence evidence, workaround, regression status. -->

## Environment

<!-- Product/build or commit, test date, local/staging/production URL, browser/OS, relevant configuration and live/mock mode. Record unknown versions honestly.
     For skeleton-maker: `skeleton-maker --version` (or the commit), Python and OS versions, the first line of `ffmpeg -version`,
     and the input video's codec, resolution and frame rate:
     ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate -of csv=p=0 <file>
     Say whether the NIM call was live or stubbed. -->

## Preconditions

<!-- Starting state, required records, access/configuration, and exact synthetic input. Include setup commands or a stable setup link when needed. -->

## Steps to reproduce

<!-- Numbered actions with exact labels/inputs and the point where the defect becomes visible. Distinct variants may have separate numbered lists. -->

## Expected behavior

<!-- What the user should observe and retain. -->

## Actual behavior

<!-- What was observed, verbatim error if any, state after waiting for completion, and scope of reproduction. -->

## Evidence

<!-- Caption and embedded screenshot(s) for UI findings; concise logs/repro artifact for nonvisual findings. State console/network coverage and gaps.
     Remove secrets first: never paste NVIDIA_API_KEY, any `nvapi-` token or an environment dump, and redact gRPC metadata and request headers in logs.
     Do not attach video you lack the rights to share; describe it or reproduce with a synthetic clip, e.g. ffmpeg -f lavfi -i testsrc=duration=2:size=320x240:rate=15 clip.mp4 -->

## Acceptance criteria

<!-- Unchecked pass/fail criteria covering the defect, correction/recovery, and relevant adjacent behavior. -->

## Scope and developer notes

<!-- Observed facts versus hypotheses, pinned code links, suggested approach without prescribing unproven internals, exclusions, related issues, and remaining unknowns. -->
