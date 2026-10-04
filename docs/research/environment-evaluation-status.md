# Environment evaluation status

Issue #51 remains subject to real-video recognition, tracking and independent geometry gates. Offline fixtures establish integration behavior. The observations below are development evidence, not held-out accuracy results.

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

## Candidate measured references

Official releases offer possible complementary scene references: [ScanNet](https://github.com/ScanNet/ScanNet) documents indoor RGB-D sequences, camera poses, meshes and projected instance annotations; [SceneNN](https://github.com/hkust-vgd/scenenn) documents indoor RGB-D scenes, intrinsics, camera trajectories and annotations; [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) documents outdoor imagery, laser scans, calibration, poses and instance labels. Their access and use terms differ: ScanNet requires an agreement, SceneNN limits its documented permission to educational/research use unless the authors are contacted, and KITTI-360 requires registration and states CC BY-NC-SA 3.0 terms. No benchmark assets were downloaded or agreements completed. Task-specific label mapping, frame/reference alignment, selected-scene coverage and independent second review remain unresolved. This shortlist does not complete the required corpus.

## Local evidence

The integration checkout's ignored `work/jev-run/` directory contains source-search metadata, download logs, full Jev envelopes and evidence, the street manifest and mask bundle, runtime logs and inspected frames. Version 2 tracking artifacts are under `work/jev-run/issue-61-real/run-3/`; the DA3 smokes, derived clip and original manifests are retained under `work/jev-run/issue-59-real/`, with a hash-checked relocation ledger. Media and model weights remain outside Git. The committed inventory and this report retain readable provenance and the decisive execution/failure observations; they do not substitute for the complete evaluation artifacts required by #66.
