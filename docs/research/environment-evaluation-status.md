# Environment evaluation status

Issue #51 remains subject to real-video recognition, tracking and independent geometry gates. Offline fixtures establish integration behavior. The earlier observations below are development evidence. The final corpus-v2 section reports the frozen development and held-out pilot, including failed gates and a missing attempt.

## Source acquisition

On 2026-10-04, 12 YouTube-origin candidate excerpts were downloaded outside Git to `/Users/Nicolas/Documents/datasets/skeleton-maker/environment-youtube-development/`. [The source inventory](environment-source-candidates.json) records original watch URLs, uploader attribution, reported license, requested excerpt intervals and actual decoded timing, local filenames, SHA-256 digests, decoded dimensions/timing and retrieval tools. It contains six indoor and six outdoor candidates. Its split is intentionally unassigned; capture-session independence, category coverage, manual masks, identity labels, second review and measured geometry are pending.

The YouTube searches used the Creative Commons filter. Source metadata reports Creative Commons Attribution reuse permission for the selected creator videos. The White House kitchen preparation video uses the reviewed public-domain record on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:In_the_Kitchen_before_the_Korean_State_Dinner.webm). Retain uploader attribution and original source URLs when sharing derived excerpts. A metadata license record is source evidence, not independent proof that every third-party element was cleared.

Selection used separate Jev screening, relevance ranking, source-type classification, evidence verification, extraction audits and bounded choices. Animation, explicit architectural 3D plans, celebrity compilations with unclear source footage, and equirectangular 360 video were excluded. Some judgments disagreed: an initial classifier accepted a 360 interior despite the stated projection exclusion, and an office-design candidate remained visually ambiguous. The source evidence and pixel inspection governed the final inventory. Jev does not see video pixels and is never a ground-truth annotator.

The Dutch apartment excerpt follows the source's ceiling and floor chapters at 04:13 and 04:43. Short traffic clips use intervals inside their actual durations. Failed download attempts remain in the local evidence logs; the final inventory lists only successfully downloaded, decoded files. Requested source windows may be shortened at end-of-file or rounded to encoded frames; the decoded duration and rational frame rate identify the actual clip clock. The exact local copies are identified by hash. Re-downloading and re-encoding a changing YouTube stream may produce different bytes, so compare hashes before using a replacement as the same evaluation input.

No six/six held-out split has been claimed. No independent annotation or reviewer identity has been invented. No source supplies the independent camera controls and withheld metric measurement required by #64 merely by being a room or driving video.

## Annotation preparation

The local candidate packet at `/Users/Nicolas/Documents/datasets/skeleton-maker/environment-youtube-development/annotation-proposal-20261004/` contains 120 PNG frames, ten per candidate, and `frame-proposal.json`. Selection uses the midpoint of ten equal source-frame strata and does not consume predictions. The ledger records exact rational timestamps, source hashes, image hashes, dimensions and the inventory hash. All 12 source hashes and 120 image hashes were checked, and each decoded frame count matched the inventory. The ledger SHA-256 is `ea69d893ddb905b59e47242880e07810bec21fb3c33f76cb21ced136b467ab17`.

This is an unreviewed frame proposal with an unassigned split. It contains no ground-truth masks or identity labels. Candidate images were inspected during source selection, and kitchen/street development inference exists; a reviewer must audit prior exposure before freezing any held-out selection. Independent annotations, second review, shot-local continuous tracking intervals, category/capture-session coverage, camera control observations and withheld measured dimensions remain required. The packet advances preparation for #64 without completing it.

## Real perception execution

The existing Grounding DINO tiny + SAM 2.1 small worker ran through the public CLI on an Apple M5 with 16 GiB system memory, using MPS. The source was a four-second excerpt of [Street traffic](https://www.youtube.com/watch?v=_7sCYyxw4Ic), with reviewed [CC BY 3.0 provenance](https://commons.wikimedia.org/wiki/File:Street_traffic.webm), conformed to 480 × 270 pixels at 10 fps. Sampling at 0.5 fps processed source frames 0 and 20.

```text
environment: complete
  entities : surface 3, vehicle 8
  frames 0-20 (2 scanned)
  geometry : not_requested (--geometry off)
```

Observed worker statistics: 52.8 seconds wall time, 991 MB peak process RSS, and 4,660 MB MPS driver allocation. Process RSS alone omits accelerator allocation. This includes loading the models and is one small run, not a throughput or memory guarantee for longer clips.

The run used Grounding DINO revision `a2bb814dd30d776dcf7e30523b00659f4f141c71` (weights SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`) and SAM 2.1 small SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`. The recorded libraries were torch 2.14.1, torchvision 0.29.1, transformers 5.18.0 and sam2 1.1.0. This exercised perception, not DA3 geometry. The manifest records checkpoint terms, device, preprocessing, settings and source hash.

## Real video tracking execution (#61)

Worker v2 ran through the public CLI on the same 4-second street-traffic development excerpt, conformed to 480 × 270 at 10 fps. The source hash was `38c01a6a74ed978f3132bd34865ead28fb1b89dad99f0694bdb73e9e69c4d517`. With `--sample-fps 1 --device mps --geometry off --no-cache`, source frames 0, 10, 20 and 30 were processed. The complete manifest contains 35 shot-scoped entities and 44 visible masks. Motion fields are unknown for 29 entities, dynamic for 5 and static for 1; the motion evidence records the thresholds as uncalibrated.

Readable examples from that manifest:

```text
shot-0/bus-20  bus  observed frames [10, 20, 30]  no gap
shot-0/bus-21  bus  observed frame [10]  lost at sampled frames [20, 30]
shot-0/car-9   car  dynamic  observed frames [0, 10]  lost at sampled frames [20, 30]
```

The bus detections from frame 10 retain distinct entity IDs. The car track records an open gap with no fabricated observations at frames 20 and 30; its reason says the detector did not confirm the propagated mask and that occlusion cannot be distinguished from a miss. There was no reacquisition in this final short run. Since the clip contains one shot, this execution did not exercise a camera cut. It used Grounding DINO tiny and SAM 2.1 small on MPS; worker statistics were 65.6 seconds, 945 MB peak process RSS and 4,637 MB MPS driver allocation (the whole command took 67.2 seconds). SAM2 warned that its optional `_C` extension was unavailable and skipped optional post-processing. Geometry was off, so this run did not test DA3 or its dynamic-pixel exclusion.

These are readable development observations from one clip, not ground truth. They do not establish identity accuracy, motion accuracy, or usable motion thresholds. The final manifest and masks are retained only under the ignored `work/issue-61-real/run-3/` directory on the development machine; no media or model assets are committed.

## Observed recognition failure

The first outdoor street frame has no visible indoor ceiling. The worker nevertheless predicted a ceiling mask covering 19.5% of the frame, including vehicle and background pixels. The rendered mask overlay was visually inspected. This is a development failure and a reason to keep the real accuracy gates open; it is not a measured held-out false-positive rate.

No precision, recall, structural IoU, IDF1, registration coverage, reprojection error or metric-scale accuracy target is claimed achieved. The [scorer](../environment-scoring.md) must consume independently reviewed annotations before those claims can be evaluated. Targets must not be silently lowered to accommodate a failed run.

## Tracking worker development run

The version 2 perception worker was exercised on the same already-exposed four-second street clip at 1 fps: source frames 0, 10, 20 and 30. Its final bounded run reported 65.6 seconds wall time, 945 MB peak process RSS and 4,637 MB of MPS driver allocation at the recorded snapshot. It used the same pinned Grounding DINO and SAM checkpoints. The optional SAM 2 `_C` extension was unavailable. This run exercises propagated masks and association, but contains no confirmed reacquisition and does not establish identity or motion accuracy, shot-cut handling, or geometry exclusion.

The negative-case failure remains. The final run labels 44,803 of 129,600 pixels in frame 0 as ceiling (34.57% of image area). Inspection of the actual source and mask shows that the region covers much of the central truck and buses in an outdoor scene. This is a qualitative development failure and an exact predicted-mask area, not an independently scored held-out false-positive rate. Both the earlier run and this newer result remain in local evidence.

## Depth worker development run

The pinned DA3 Small worker was exercised on MPS with 16 frames from a derived 640 × 360, 30 fps cooking clip, in one shot. It produced all 16 native depth assets. The public invocation used deterministic fake semantic responses with the actual depth model; it is a geometry smoke, not a combined real perception evaluation. Recorded worker wall time was 1.937 seconds. Its timer starts after heavy imports and before model loading, so it includes model loading and inference but omits interpreter startup and imports. Peak process RSS was 732,741,632 bytes. Synchronized post-inference MPS tensor and driver allocations were 156,486,912 and 2,321,252,352 bytes. These are GPU allocation snapshots, not peak GPU measurements. The worker advertises a 16-frame request limit; this bounded run does not establish a larger batch or cross-device envelope.

A separate five-frame DA3 run used the existing cooking pose file and excluded supplied NIM skeleton boxes from the depth grid. The pose file is legacy JSON Lines without a source fingerprint, so its source association remains user-supplied. The results remain relative camera-frame geometry with no solved metric registration. DA3 code revision was `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`; the DA3 Small model revision was `e08cab65ca0ec38e7826075418411ab90cab4da3`, with checkpoint SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.

## Combined real development execution

At source commit `4ea87e1da74affcb89748ba9ca578850311a01df`, the public `environment` command ran actual Grounding DINO, SAM 2.1 video tracking and DA3 Small together on the existing 20-second cooking clip (640 × 360, 30 fps), with `--sample-fps 0.2 --device mps --geometry auto --no-cache`, supplied poses, an overlay and a viewer. It processed source frames 0, 150, 300 and 450, producing 17 predicted entities and 46 observations. The command exited 0 after 78.677 seconds: perception was complete, while the overall manifest was partial because geometry remained `relative-camera-frame` without independent controls or solved metric scale. No hosted NIM request was needed: the existing 600-frame pose file was reused. Its legacy records have no source fingerprint, so association remains explicitly user-supplied.

Perception worker statistics were 68.5 seconds, 810 MB peak process RSS and a 4,637 MB MPS driver-allocation snapshot. DA3 produced four native 280 × 504 depth arrays; its recorded timer was 1.278 seconds, starting after heavy imports and before model loading. DA3 peak process RSS was 708,411,392 bytes; synchronized post-inference tensor and driver allocations were 143,902,976 and 1,419,493,376 bytes. Accelerator allocations are snapshots, not peak measurements. This bounded development run does not establish a throughput or memory envelope for longer clips.

The output audit verified 103 recorded artifact hashes and both original input hashes. The overlay retained 600 frames, 640 × 360 pixels, 30 fps and a 20-second duration. All pixels selected by the worker's OpenCV nearest-neighbor exclusion mapping were invalid in the emitted depth arrays. A stronger audit using exact rational source-grid coordinates differed at integer row boundaries and found 347, 5, 2 and 10 finite pixels selected by that alternative mapping; the discrepancy is retained for review. These checks establish artifact behavior, not independent recognition or geometry accuracy.

The actual local viewer passed decoded-source pixel comparisons at frames 0, 1 and 150, family visibility controls, unsampled pose rendering and local-asset/runtime checks. The original OpenCV reference comparison failed; explicitly decoding the source's declared BT.709 values resolved that mismatch, and raw RGB references avoided a separately measured browser transformation of tagged PNG images. Both failed comparisons remain in the evidence. The original tolerances were retained: mean absolute RGB-channel error at most 2 and at most 0.1% of channels differing by more than 8. Final mean errors were 0.4375, 0.4379 and 0.4188, respectively, with no channels exceeding 8. The viewer reports relative geometry; predictions shown in its screenshot are not reviewed annotations.

The exact invocation, source and code hashes, hardware, manifest, masks, native depth arrays, overlay, browser script/logs and screenshot are retained under the integration checkout's ignored `work/jev-run/combined-real-development/` and `work/resume-jev/` directories. The execution belongs to the source commit named above; subsequent source changes must be checked separately. Neither the independent corpus gate (#64) nor the frozen held-out scoring gate (#66) is complete.

## Final source development verification

The clean consolidated source commit `84f0d69d68f488c4487e0b1572d792a6e2c1e9d8` (tree `e8bc82ae7a13e9aebaef16e191d37162524278c8`) was checked separately on 2026-10-04. Two complete `just check` runs each passed 485 tests with one skip and 82.67% coverage, including shipped worker code; only generated gRPC stubs are omitted. The coverage floor is 82. Offline browser checks exercise two distinct registered environment shots, exact source-frame mapping, and hidden geometry when a matching camera transform is absent. Public regressions cover nonzero camera skew with Brown-Conrady distortion, agreement among the worker image, exclusion mask and calibration pixels, prior zero-skew image/mask parity, and distortion-free skew preservation. These camera checks use known synthetic geometry, not independently measured real-scene controls.

The actual pinned Grounding DINO, SAM 2.1 video and DA3 Small pipeline was rerun from that clean commit on the same 20-second cooking source and supplied poses, using the same public command options as the historical combined execution above. It exited 0 in 74.244 seconds, sampled frames 0, 150, 300 and 450, and emitted 17 entities and 46 observations. Perception was complete; the run remained partial with relative camera-frame geometry and no solved metric scale. No NIM call was needed. The legacy pose association remains user-supplied.

The perception worker recorded 65.6 seconds, 838 MB peak process RSS and a 4,629 MB MPS driver-allocation snapshot. DA3's timer began after heavy imports and before model loading and recorded 0.823 seconds, with 709,246,976 bytes peak process RSS. Its synchronized post-inference tensor and driver allocations were 143,902,976 and 1,419,493,376 bytes. These accelerator values are snapshots, not peaks. Four native depth arrays were 280 × 504. This bounded run does not establish a larger throughput or memory envelope.

The final artifact audit verified all 103 pipeline output hashes and both original input hashes. The overlay retained 640 × 360 pixels, 600 frames, 30 fps and 20 seconds. All excluded depth pixels under the documented legacy OpenCV nearest-neighbor raster mapping were invalid. The separately retained exact rational-grid diagnostic still found 347, 5, 2 and 10 finite pixels selected by that different mapping; it has not been relabeled as a successful rational-grid check.

The first fresh browser audit timed out before pixel comparison. Diagnostics showed a requested frame 1, but Chromium reported a seekable range of `0..0` and decoded frame 0: the audit's generic HTTP server did not serve byte ranges. Reusing the repository's existing range-aware loopback server resolved this audit setup failure without a production change. The original failure and instrumented traces remain retained. With the original tolerances unchanged, decoded frames 0, 1 and 150 passed raw BT.709 source comparisons with mean RGB-channel errors 0.4375, 0.4379 and 0.4188, respectively, and no channels exceeding an error of 8. Family controls, selected-entity evidence, unsampled pose rendering and local-asset/runtime checks passed; the screenshot was visually inspected as a development prediction display.

The public `character` command also consumed the fresh manifest, original pose file and original video. It generated the 600-frame stage after source hash and clock validation, while exposing relative geometry with empty environment point clouds. It did not display relative depth as registered metric geometry. A separate fixed-gzip-clock 20-frame public fixture remained byte-identical to the pre-environment-stage baseline.

The fresh outputs, exact invocation, source and tracked package hashes, runtime statistics, raw RGB references, failed and successful browser logs, screenshot and character artifact audit are retained under ignored `work/jev-run/final-combined-real-development/`. Historical outputs remain separate. Independent reviews found no remaining concrete software mismatch, while Jev patch gates retained confidence and test-gap escalations; those judgments are advisory, not accuracy certification. The independent corpus and held-out evaluation gates (#64 and #66) remain incomplete, and #51 remains open with PR #67 draft.

## Candidate measured references

Official releases offer possible complementary scene references: [ScanNet](https://github.com/ScanNet/ScanNet) documents indoor RGB-D sequences, camera poses, meshes and projected instance annotations; [SceneNN](https://github.com/hkust-vgd/scenenn) documents indoor RGB-D scenes, intrinsics, camera trajectories and annotations; [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) documents outdoor imagery, laser scans, calibration, poses and instance labels. Their access and use terms differ: ScanNet requires an agreement, SceneNN limits its documented permission to educational/research use unless the authors are contacted, and KITTI-360 requires registration and states CC BY-NC-SA 3.0 terms. No benchmark assets were downloaded or agreements completed. Task-specific label mapping, frame/reference alignment, selected-scene coverage and independent second review remain unresolved. This shortlist does not complete the required corpus.

## Local evidence

The integration checkout's ignored `work/jev-run/` directory contains source-search metadata, download logs, full Jev envelopes and evidence, the street manifest and mask bundle, runtime logs and inspected frames. Version 2 tracking artifacts are under `work/jev-run/issue-61-real/run-3/`; the DA3 smokes, derived clip and original manifests are retained under `work/jev-run/issue-59-real/`, with a hash-checked relocation ledger. Media and model weights remain outside Git. The committed inventory and this report retain readable provenance and the decisive execution/failure observations; they do not substitute for the complete evaluation artifacts required by #66.

## Frozen corpus-v2 evaluation

The once-only pilot failed the frozen accuracy gates. Seven development clips and eight held-out clips were attempted. 14 prediction manifests were published; their individual perception, overall and geometry statuses are retained in the JSON report. Held-out `vip-h3-indoor-objects` stopped after 12 of 16 samples without a published manifest; its termination cause, exit status, wall time and peak RSS were not captured. Its expected command is reconstructed from the freeze, with provenance explicitly marked because the original runtime argv was not captured. The interrupted attempt was retained and was not retried. Missing predictions count as misses, so these are failed full-pilot results, not a completed-cohort accuracy estimate.

The [machine-readable results](environment-corpus-v2-results.json) include every scorer gate, per-class and per-clip results, exact frozen commands/settings/aliases/targets, worker identities, hashes, hardware and per-clip execution records. Evaluated source commit: `cb8f2749cab3d3e3dbf2c09821bee78a149fe817`. The original configuration was frozen before held-out inference at `2026-10-05T11:39:30.375233+00:00`; its SHA-256 is `2e80a8ae6ae008b172ee3c5c64d7b55e36ede248264ac11b07adb66ac4073bca`. The 2,506 corpus inputs retain tree SHA-256 `558ae101cea0c23ddb7e1dfa36151c487f6eb95e5b13729377d6b921f8813d2a`. The original full hash inventory, media, annotations, masks, depth and logs remain outside Git under `/Users/Nicolas/Documents/datasets/skeleton-maker/corpus-v2`.

CPU was selected from development resource/execution evidence before accuracy scoring. The MPS attempt was operator-stopped after 1,257.8 seconds amid swap and slow progress; no deadlock was proven. The completed CPU d1 probe was copied with identical hashes into the final prediction directory, preserving original command/log paths in its record. The baseline aliases, targets and inference settings were not tuned after development error analysis or held-out exposure. Hardware: Apple M5, 16 GiB, arm64, macOS 26.4.1.

At the evaluated source commit `cb8f274`, the public scorer rejected the raw runner directory with exit 2 because run-record JSON was parsed as a prediction. Subsequent Qodo review fixes allow direct scoring by ignoring only known runner/operator metadata sidecars; the frozen pilot scores below retain the original scorer and audited manifest-only view. The frozen workaround uses a manifest-only symlink view: every manifest and asset is validated, runner/operator-stop metadata exclusions are recorded in a separate ledger, and score files are written outside both input directories. The scorer and targets were unchanged.

### Separate gate results

Both final public scorer invocations exited 1. Accuracy is split-specific; corpus coverage examines all 15 annotations. There is no combined score. Full reasons, subtype counts and per-clip failures are in the JSON report.

| Gate | Development | Held-out |
| --- | --- | --- |
| objects | FAIL; P/R 0.293185/0.435294; TP/FP/FN 185/446/240 | FAIL; P/R 0.592697/0.241972; TP/FP/FN 211/145/661 |
| vehicles | FAIL; P/R 0.625731/0.295580; TP/FP/FN 107/64/255 | FAIL; P/R 0.739623/0.532609; TP/FP/FN 196/69/172 |
| walls | FAIL; IoU 0.212924; negative FP 37/47 (0.787234) | FAIL; IoU 0.300508; negative FP 17/48 (0.354167) |
| floors | FAIL; IoU 0.759878; negative FP 43/47 (0.914894) | FAIL; IoU 0.334738; negative FP 47/48 (0.979167) |
| ceilings | FAIL; IoU 0.444931; negative FP 60/63 (0.952381) | FAIL; IoU 0.853950; negative FP 37/48 (0.770833) |
| structural_surfaces | FAIL; AND of wall, floor and ceiling gates | FAIL; AND of wall, floor and ceiling gates |
| identity_objects | FAIL; IDF1 0.317959; switches 39, fragmentation 13, misses 235 | FAIL; IDF1 0.270358; switches 47, fragmentation 12, misses 661 |
| identity_vehicles | FAIL; IDF1 0.307692; switches 26, fragmentation 8, misses 255 | FAIL; IDF1 0.565561; switches 19, fragmentation 8, misses 172 |
| registration | FAIL; 0/4 registered; reprojection error unmeasurable | FAIL; 0/8 registered; reprojection error unmeasurable |
| metric_scale | FAIL; no predicted check dimensions | FAIL; no predicted check dimensions |
| underconstrained | PASS | FAIL; missing h3 manifest cannot establish honest abstention |
| corpus_coverage | FAIL; failed: complete prediction runs | FAIL; failed: complete prediction runs |

### Execution records

| Clip | Split | Exit | Wall seconds | Peak process RSS bytes | Result |
| --- | --- | --- | --- | --- | --- |
| tum-fr1-rpy | heldout | 0 | 1444.8 | 7188561920 | perception complete; overall partial |
| tum-fr1-xyz | heldout | 0 | 1792.6 | 6872023040 | perception complete; overall partial |
| tum-fr2-rpy | development | 0 | 2422.6 | 6618284032 | perception complete; overall partial |
| vip-d1-indoor-structure | development | 0 | 4669.3 | 3900293120 | perception complete; overall partial |
| vip-d2-indoor-structure | development | 0 | 2592.3 | 6516867072 | perception complete; overall partial |
| vip-d3-indoor-objects | development | 0 | 1149.9 | 4939235328 | perception complete; overall partial |
| vip-d4-vehicles | development | 0 | 1574.5 | 6176505856 | perception complete; overall partial |
| vip-d5-vehicles | development | 0 | 1886.6 | 6051905536 | perception complete; overall partial |
| vip-d6-vehicles | development | 0 | 1241.1 | 4656742400 | perception complete; overall partial |
| vip-h1-indoor-structure | heldout | 0 | 4897.5 | 6850297856 | perception complete; overall partial |
| vip-h2-indoor-structure | heldout | 0 | 3251.5 | 3493494784 | perception complete; overall partial |
| vip-h3-indoor-objects | heldout | unknown | unknown | unknown | interrupted; no manifest |
| vip-h4-vehicles | heldout | 0 | 3971.0 | 3164635136 | perception complete; overall partial |
| vip-h5-vehicles | heldout | 0 | 2693.1 | 3181150208 | perception complete; overall partial |
| vip-h6-vehicles-fixed | heldout | 0 | 2766.4 | 6170705920 | perception complete; overall partial |

Process RSS excludes accelerator allocation and is not total physical footprint. The machine was shared with other active workloads, so these runtimes are not a throughput benchmark. CLI exit zero certifies neither recognition accuracy nor metric registration. The optional SAM2 `_C` extension warning remains in the logs.

### Development diagnostics and limits

The read-only development mask audit reproduces object and vehicle counts. Of 446 unmatched object predictions, 96 have best same-class IoU below 0.5, 44 overlap an eligible same-class reference but remain unmatched by one-to-one assignment, and 306 have no same-class reference in that frame. The 64 unmatched vehicle predictions divide into 23, 1 and 40 respectively. Existing chair/stool and table/desk aliases cover those terms. Duplicate masks, ambiguous `shelf cabinet` labels and distinct bed/sofa or truck/car classes were retained. High overlap with a differently labeled published instance does not establish an alias. The preselected d1 frame-zero ceiling diagnostic has IoU 0.247543: predicted outlines miss broad published ceiling regions.

The held-out underconstrained gate also fails for the absent h3 manifest: its geometry status is missing and therefore not in the honest-geometry set used by the scorer. The raw generic reason says that the clip claims registration without adequate evidence; no actual registration claim by this missing clip was observed. The reason string is preserved in JSON.

Structural negative rates measure agreement with the frozen VIPSeg mapping. Ground/road/building/sky classes remain background in that mapping, so negative failures do not establish that every predicted physical surface is semantically wrong. The missing held-out clip contributes positive misses but its unscanned class-negative frames do not enter the negative-rate denominator, as specified by the scorer.

Development TUM registration abstained: ten of 20 supplied anchors lack positive finite DA3 depth and ten have camera-intrinsics mismatch; none passes. Frame-zero returned focal lengths are 611.926/610.084 versus corrected calibration 544.841/543.589. Held-out TUM fr1-rpy abstains for nonpositive/nonfinite anchor depth; fr1-xyz abstains for intrinsics mismatch. Guards and controls were preserved. This records rejection conditions without asserting a fix. Check dimensions are withheld from scale fitting but their expected lengths appear in the calibration payload for post-fit validation, so the payload is not fully blinded.

VIPSeg published quality control replaces a newly named second reviewer under the accepted deviation. Training contamination is unknown. Pan occurs only in development; the blur proxy was checked on one frame pair. TUM room/360 references were rejected for depth/pose disagreement. Freiburg3 sitting rotation qualified by a fragile 0.0001 m margin and was rejected during selection. Results apply only to this finite pilot and fixed taxonomy.

A fresh `just check` at the evaluated source commit passed: 532 tests, one skip, 82.76% coverage, plus hooks, format/lint/types, workflow security, research preflight and all offline browser regressions. Report-only changes are checked separately by the repository hooks. Fixture success establishes software behavior, not recognition accuracy. All twelve Jev capabilities were used with `typesafe/jev-1.13` through OpenRouter; local envelopes retain contradictions and low-confidence/escalated judgments. Actual numerical checks and artifact validation supply proof. Issue #51 and the corpus/evaluation tickets remain open, and PR #67 remains draft. No merge or issue closure is claimed.

The final Jev report gate verified all five completion claims, with no contradicted or unsupported claims, but escalated its patch review: safe-to-apply 0.40, correctness confidence 0.46 and blast-radius confidence 0.41. Hash/report equivalence was verified with review confidence 0.76. This is not automatic approval. The stronger agent review resolves report-only safety from the exact four-file diff and the executed corpus/source hash, raw-score equality, manifest/record and hook checks; no numerical gate or parent-feature readiness is overridden. The final probabilities are retained in the JSON report, and the initial oversized-request API 400 remains in local evidence.
