# Environment worker: Grounded SAM 2

The base package contains no model code. The real perception backend, `grounded-sam2-da3`, runs as a separate worker in its own virtualenv and is called over the worker protocol described in `skeleton_maker/envworkers.py`. This page records what was set up and exercised, the exact pins and the model terms.

## Set up

```bash
just environment-worker-setup
```

This runs `scripts/setup_environment_worker.py`. It creates `~/.cache/skeleton-maker/workers/grounded-sam2-da3/.venv` (Python 3.12), installs the exact pins from `skeleton_maker/workers/grounded_sam2_da3.requirements.txt`, downloads the two model files into `~/.cache/skeleton-maker/models`, and checks each against its pinned SHA-256. It is safe to run again. Override the locations with `SKELETON_MAKER_WORKER_HOME` and `SKELETON_MAKER_MODELS`. Nothing is written inside the repository.

About 1 GB is downloaded in total (the libraries come to roughly 2 GB on disk with PyTorch).

## Pins and terms

| Part | Pin | Terms |
| --- | --- | --- |
| PyTorch / torchvision | `torch==2.14.1`, `torchvision==0.29.1` | BSD-style, see PyPI |
| Transformers | `transformers==5.18.0` | Apache-2.0 |
| SAM 2 code | `sam2==1.1.0` (official `facebookresearch/sam2` package) | Apache-2.0 |
| SAM 2.1 weights | `sam2.1_hiera_small.pt`, SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`, from `dl.fbaipublicfiles.com/segment_anything_2/092824/` | Apache-2.0 |
| Grounding DINO weights | `IDEA-Research/grounding-dino-tiny` at revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, `model.safetensors` SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` | Apache-2.0 (model card) |

Preflight refuses to run when an installed library differs from its pin, so a changed library cannot silently change results. The full list of installed packages is in the requirements file. Check the terms yourself before any commercial use; they were read from the model cards and repositories listed above.

## How a run works

For each scanned frame the worker prompts Grounding DINO with the request's label vocabulary (the preset plus `--classes`, see [environment.md](environment.md#labels)): `wall`, `floor` and `ceiling` one at a time, every other label in groups of four, with box threshold 0.25 and text threshold 0.2. Boxes are de-duplicated per matched label (IoU 0.7, at most 6 per object label, 4 `unknown` boxes per frame). The worker stages the sampled frames into a temporary folder per visual shot and initializes a fresh SAM 2.1 video predictor state for each shot. Grounding DINO runs on every sample to find new entrants. Current boxes are associated one-to-one only with same-label, prior SAM2-propagated masks using detector-box pixel coverage, a minimum 0.35 overlap and 0.12 ambiguity margin; those scores are explicit heuristics, not identity probabilities. Unmatched boxes seed new predictor objects. People are never reported: the skeletons own them.

- Surfaces become one `surface` entity per class per shot; each sample's boxes for that class are merged to seed one propagated mask.
- Objects and vehicles keep separate shot-scoped entity IDs when the propagated mask evidence supports association. A track stores only confirmed visible masks. Missing sampled frames are listed exactly in `track.gaps.sampled_frame_ids`; an empty SAM2 mask does not produce a visible observation. Reacquisition is recorded only after a later non-empty mask, and association candidates, score and meaning remain visible in the manifest.
- Motion is estimated without the semantic class: sampled detector-box centers are compared after subtracting median full-frame optical flow. Motion stays `unknown` when there are too few observations or the camera-flow dispersion exceeds its threshold. The displacement and dispersion thresholds are heuristic and are marked `calibrated: false` in each entity's `motion_evidence`.
- `static` motion entities are eligible for downstream static geometry. `dynamic` and conservative `unknown` entities remain in the manifest but are excluded from the geometry adapter's static fusion inputs. This is not a measured motion-accuracy result.
- Scores are Grounding DINO box confidences (uncalibrated); SAM2 masks and association scores are not confidence-calibrated.
- The raw, pre-de-duplication detections of every frame are kept in `<manifest stem>.assets/raw/detections.json` next to the settings that produced them.
- The manifest `backend` block carries model and checkpoint identity, library versions, settings and `run_stats` (wall seconds, peak process resident memory, and an end-of-run MPS driver-allocation snapshot when running on MPS).

Options and limits: `--sample-fps` bounds the work, and the geometry worker preflight advertises a maximum batch of 16 sampled frames. In `auto` mode, a larger request keeps the semantic result and reports geometry unavailable with the frame-limit reason. In `required` mode it rejects the request before semantic inference. It does not silently split a shot into batches because each shot must retain one DA3 camera reference frame. A frame that cannot be decoded makes the run `partial` (exit 3) and names the frame ids; any other failure fails the run and writes nothing.

## Cache

Inference results are cached under `~/.cache/skeleton-maker/environment` (or `--cache-dir`; `--no-cache` turns it off). The key is a SHA-256 of the clip's hash, scanned frames and shots, the exact supplied pose-file hash, the requested labels, geometry mode, semantic and geometry devices, the validated calibration reference, and each worker's identity (models, checkpoint hashes, library versions, preprocessing and settings). Changing any of them runs the worker again. A cache entry is revalidated on every hit; a damaged or tampered entry is ignored and replaced. Only complete runs are cached.

## What was exercised

| | |
| --- | --- |
| Device | Apple M5, 16 GB unified memory, macOS (Darwin 25.4), arm64, PyTorch MPS |
| Clip | `in/cooking.skeleton/clip.mp4`: 640x360, 30 fps, 600 frames; `--sample-fps 0.5` (10 frames), reusing the clip's poses, `--geometry off` |
| Result | complete; 3 surface entities, 63 object detections (54 `bottle`, 9 `cup`), 93 asset files |
| Wall time | 107 s inside the worker, 111 s for the whole command (about 11 s per scanned frame, including loading the models) |
| Memory | 923 MB peak resident for the worker process; 4.7 GB driver-allocated MPS memory at the end of the run |
| Second identical run | cache hit, no worker call |

The tracking worker was also run through the public CLI on a four-second street-traffic excerpt at 480 × 270, 10 fps, with `--sample-fps 1 --device mps --geometry off --no-cache`. It processed source frames 0, 10, 20 and 30 from SHA-256 `38c01a6a74ed978f3132bd34865ead28fb1b89dad99f0694bdb73e9e69c4d517` using worker version 2. The complete manifest contains 35 entities and 44 visible observations: 29 unknown, 5 dynamic and 1 static entity. For example, `shot-0/car-9` has observed masks at frames 0 and 10 and an open lost gap at sampled frames 20 and 30; the record says detector non-confirmation cannot distinguish occlusion from a missed detection. This illustrates persistent identities and explicit gaps, without claiming that the predicted IDs or motion labels are correct. This bounded run had open gaps and no reacquisition.

| | |
| --- | --- |
| Worker wall time | 65.6 s; complete CLI wall time 67.2 s |
| Peak process RSS | 945 MB |
| MPS driver allocation | 4,637 MB |
| Assets | 45 files, including masks and raw detector evidence |
| Geometry | Off; this run did not execute DA3 or verify geometry exclusion |

The video predictor logged that its optional SAM2 `_C` extension was unavailable and skipped its optional post-processing step. The real run exercised video propagation on MPS despite that warning. Its outputs remain development evidence only; accuracy and motion thresholds require independent annotation and evaluation.

This was a single clip on a single machine. The only device exercised is MPS. **CPU, CUDA, Linux and other Macs are untested**, although preflight lists `cuda` and `cpu` when PyTorch reports them. Other model sizes were not tried.

Looking at the masks on frame 120 of that clip, the bottles near the sink are found, but the `floor` and `ceiling` regions are noisy and spill over cabinets and walls. That is an observation from one frame, not a measurement. Recognition accuracy is only established by scoring against human labels (`environment-score`, see [environment-scoring.md](environment-scoring.md)); no such evaluation has been run.

## DA3-Small geometry worker

Install the optional depth and camera worker with:

```bash
just da3-geometry-setup
```

It uses a separate Python 3.12 environment at `~/.cache/skeleton-maker/workers/da3-geometry/.venv`; the model and pinned DA3 source are cached under `~/.cache/skeleton-maker/models`. Set `SKELETON_MAKER_WORKER_HOME` or `SKELETON_MAKER_MODELS` to change those locations. Its `--geometry auto` path batches sampled images by detected shot. Each shot receives an independent camera reference frame; poses use DA3/OpenCV world-to-camera extrinsics. Depth assets retain DA3's native processed pixel grid. The manifest records actual source and depth dimensions and independent x/y source-to-depth and inverse transforms. Intrinsics are converted to undistorted source-frame pixels using those measured resize ratios; a lens transform separately records any OpenCV undistortion.

The worker reports `relative-camera-frame`, `relative_depth`, and `metric: false`. The optional measured anchors and dimensions are validated and recorded; this worker does not use them to solve scale or register geometry to NIM skeletons. DA3-Small checkpoint and DA3 source are licensed Apache-2.0 according to the upstream model card and repository model table. The code revision, checkpoint revision and hash, full library pins, preprocessing and settings are recorded in each geometry result.

The installed worker reports usable `mps`, `cuda` and/or `cpu` devices during preflight. Only the device in the exercise record below has been run here; other listed devices are untested.

Entity IDs whose motion is `static` and which have observations are recorded under `geometry.static_fusion_entities` as candidates for downstream static geometry. Dynamic and unknown-motion entities remain in the environment inventory but appear under `geometry.excluded_dynamic_entities`; unknown motion is conservatively excluded. For each available mask, the DA3 worker maps its source-frame pixels through lens correction and resize into the native depth grid, then stores excluded depth pixels as NaN. When an observation has no mask, its validated source-frame bounding box is excluded instead. The entity record identifies observed, mask-excluded and box-excluded frames. When `--poses` supplies NIM tracks, `geometry.excluded_skeletons` records their separate `nim-skeleton:<tracking_id>` namespace and source-frame boxes are invalidated in depth at sampled frames. Without supplied poses, `geometry.skeleton_exclusion.status` is `unavailable` with a reason. The adapter does not fuse frames into a registered point cloud or scene mesh.

### Geometry worker exercise

| | |
| --- | --- |
| Device | Apple M5, 16 GB unified memory, macOS (Darwin 25.4), arm64, PyTorch MPS |
| Clip | `in/cooking.skeleton/clip.mp4`, 640×360 at 30 fps; 5 frames sampled at 0.25 fps through the public CLI/artifact path with local deterministic semantic fixture |
| Result | `relative-camera-frame`; five native 504×280 depth assets; source/depth x and y scales recorded separately. A second run supplied one deterministic synthetic dynamic mask: all 961 transformed mask pixels were NaN in the frame-480 depth asset, with none outside the mask. |
| DA3 worker wall time | 1.321 s for the no-mask five-frame run; 1.116 s for the synthetic-mask rerun. The timer starts after worker imports and adapter setup, before request preparation and model loading; it includes model loading and frame processing, then stops before writing the response file. |
| Peak worker RSS | 637,763,584 bytes (about 608 MiB) and 624,328,704 bytes (about 595 MiB), respectively. |
| Bounded-batch probe | A real 16-frame subclip from the same local source completed as one shot and wrote 16 depth assets in 1.937 s. The timer starts after worker imports and adapter setup, before request preparation and model loading; it includes model loading and frame processing, then stops before writing the response file. Process peak RSS was 732,741,632 bytes. After-inference synchronized MPS snapshots were 156,486,912 tensor bytes and 2,321,252,352 Metal-driver bytes; these are post-inference snapshots, not peak GPU-allocation measurements. |

All runs used the deterministic local semantic fixture in place of the perception worker. The synthetic-mask run verifies coordinate mapping and stored pixel exclusion only; it says nothing about mask quality, DA3 accuracy or held-out performance. The 16-frame probe supports the installed worker's conservative per-shot batch cap on this M5/MPS machine; it does not characterize other devices or clips.

## Tests

Offline tests use a stand-in worker that speaks the same protocol (`tests/fake_worker.py`) and never touch the real models. One opt-in test runs the real worker; CI never does:

```bash
SKELETON_MAKER_REAL_WORKER=1 SKELETON_MAKER_REAL_CLIP=in/cooking.skeleton/clip.mp4 uv run pytest tests/test_environment_real_worker.py -s
```
