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

For each scanned frame the worker prompts Grounding DINO once per structural surface (`wall`, `floor`, `ceiling`), and in groups of four for the other labels (`chair`, `table`, `bottle`, `cup`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`), with box threshold 0.25 and text threshold 0.2. Boxes are de-duplicated per label (IoU 0.7, at most 6 per object label), then SAM 2.1 turns each box into a mask. People are never reported: the skeletons own them.

- Surfaces become one `surface` entity per class per shot; the frame's boxes for that class are merged into one union mask.
- Objects and vehicles become one entity per detection per scanned frame. Tracking identity across frames is not claimed; it is a later stage.
- `motion` is `unknown` for everything. Scores are Grounding DINO box confidences (uncalibrated); the mask is not scored.
- The raw, pre-de-duplication detections of every frame are kept in `<manifest stem>.assets/raw/detections.json` next to the settings that produced them.
- The manifest `backend` block carries model and checkpoint identity, library versions, settings and `run_stats` (wall seconds, peak resident memory, peak MPS memory).

Options and limits: `--sample-fps` bounds the work, and the worker refuses more than 600 scanned frames. A frame that cannot be decoded makes the run `partial` (exit 3) and names the frame ids; any other failure fails the run and writes nothing.

## Cache

Inference results are cached under `~/.cache/skeleton-maker/environment` (or `--cache-dir`; `--no-cache` turns it off). The key is a SHA-256 of the clip's hash, the scanned frames and shots, any poses, the requested labels, the geometry mode, the device and the worker's identity (models, checkpoint hashes, library versions, preprocessing, settings). Changing any of them runs the worker again. A cache entry is revalidated on every hit; a damaged or tampered entry is ignored and replaced. Only complete runs are cached.

## What was exercised

| | |
| --- | --- |
| Device | Apple M5, 16 GB unified memory, macOS (Darwin 25.4), arm64, PyTorch MPS |
| Clip | `in/cooking.skeleton/clip.mp4`: 640x360, 30 fps, 600 frames; `--sample-fps 0.5` (10 frames), reusing the clip's poses, `--geometry off` |
| Result | complete; 3 surface entities, 63 object detections (54 `bottle`, 9 `cup`), 93 asset files |
| Wall time | 107 s inside the worker, 111 s for the whole command (about 11 s per scanned frame, including loading the models) |
| Memory | 923 MB peak resident for the worker process; 4.7 GB driver-allocated MPS memory at the end of the run |
| Second identical run | cache hit, no worker call |

This was a single clip on a single machine. The only device exercised is MPS. **CPU, CUDA, Linux and other Macs are untested**, although preflight lists `cuda` and `cpu` when PyTorch reports them. Other model sizes were not tried.

Looking at the masks on frame 120 of that clip, the bottles near the sink are found, but the `floor` and `ceiling` regions are noisy and spill over cabinets and walls. That is an observation from one frame, not a measurement. Recognition accuracy is only established by scoring against human labels (`environment-score`, see [environment-scoring.md](environment-scoring.md)); no such evaluation has been run.

## Tests

Offline tests use a stand-in worker that speaks the same protocol (`tests/fake_worker.py`) and never touch the real models. One opt-in test runs the real worker; CI never does:

```bash
SKELETON_MAKER_REAL_WORKER=1 SKELETON_MAKER_REAL_CLIP=in/cooking.skeleton/clip.mp4 uv run pytest tests/test_environment_real_worker.py -s
```
